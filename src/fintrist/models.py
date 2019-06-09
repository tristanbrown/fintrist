"""
The engine that applies analyses to data and generates alerts.
"""
import logging
import pickle
import inspect
import re
from datetime import datetime as dt
from dateutil.tz import tzlocal

from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (
    DateTimeField, DictField, EmbeddedDocumentField,
    EmbeddedDocumentListField, IntField, FileField,
    ListField, MapField, ReferenceField, StringField,
)
from mongoengine.errors import DoesNotExist
from mongoengine import signals
from apscheduler.jobstores.base import JobLookupError
from bson.dbref import DBRef

from fintrist import processes, util
from fintrist.scheduling import scheduler
from fintrist.notify import Notification

__all__ = ('Stream', 'Study', 'Process', 'Trigger')

logger = logging.getLogger(__name__)

class Alerts(EmbeddedDocument):
    """Alerts generated by a Study."""
    timestamp = DateTimeField(default=dt.now(tzlocal()))
    active = ListField(StringField(max_length=120))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

class AlertsLog(EmbeddedDocument):
    """Log of Alerts."""
    log = EmbeddedDocumentListField('Alerts')
    count = IntField(default=100)

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    def trim(self):
        """Trim the log to the specified size."""
        if len(self.log) > self.count:
            self.log = self.log[:self.count]

    def record_alerts(self, alerts, timestamp):
        """Take in a new list of alerts.

        alerts (list): [str]
        """
        newalerts = Alerts(timestamp=timestamp, active=alerts)
        self.log = [newalerts] + self.log

    def get_alerts(self, idx):
        """Get the alerts at the given lookback index."""
        try:
            return set(self.log[idx].active)
        except IndexError:
            return set()

    @property
    def newactive(self):
        """Newly active alerts."""
        return self.get_alerts(0) - self.get_alerts(1)

    @property
    def newinactive(self):
        """Newly inactive alerts."""
        return self.get_alerts(1) - self.get_alerts(0)

    def clear(self):
        """Delete all alerts."""
        self.log = []

class Trigger(EmbeddedDocument):
    """A rule determining how an action is triggered."""
    alert_types = ('active', 'inactive', 'all')
    match_if = ('in', 'is',)
    action_choices = ('log', 'printhead', 'email', 'sms', 'trade')
    on = StringField(default='active', choices=alert_types)
    condition = StringField(default='in', choices=match_if)
    matchtext = StringField(max_length=120)
    actions = ListField(StringField(choices=action_choices))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    def __str__(self):
        return f"{self.matchtext} {self.condition} {self.on}"

    def check_conds(self, study):
        """Check if the conditions for triggering have been met by the alert."""
        if self.on == 'active':
            alerts = study.alertslog.newactive
        elif self.on == 'inactive':
            alerts = study.alertslog.newinactive
        elif self.on == 'all':
            alerts = study.alerts
        triggered = []
        for alert in alerts:
            if self.condition == 'in' and self.matchtext in alert:
                triggered.append(alert)
            elif self.condition == 'is' and self.matchtext == alert:
                triggered.append(alert)
        if triggered:
            self.fire(study=study, alerts=triggered)

    def fire(self, **kwargs):
        """Trigger the specified actions."""
        Notification(self.actions, **kwargs)

class Stream(Document):
    """Hosts the Studies, running them periodically.

    Usage: Stream(name, refresh)
    """
    # pylint: disable=no-member
    name = StringField(max_length=120, required=True, unique=True)
    refresh = IntField(min_value=15)
    studies = ListField(ReferenceField('Study'))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    def clean(self):
        """Clean out invalid data."""
        for study in self.studies:
            if isinstance(study, DBRef):
                self.studies.remove(study)

    def create_study(self, name, proc_name, inputs):
        """Create a new study tracked by a Stream instance."""
        try:
            self.reload()
        except DoesNotExist as ex:
            logger.error("Please save the Stream before trying to create a Study.")
            raise ex
        proc_ref = Process.objects(name=proc_name).get().id
        newstudy = Study(name, proc_ref, inputs)
        newstudy.save()
        self.add_study(newstudy)

    def add_study(self, newstudy):
        """Add an existing Study to the Stream."""
        self.update(add_to_set__studies=newstudy)
        self.reload()

    def remove_study(self, oldstudy):
        """Remove a Study from the Stream."""
        self.update(pull__studies=oldstudy)
        self.reload()

    def list_studies(self):
        """List the studies associated with this stream, by name."""
        return [study.name for study in self.studies]

    def move_study_to_idx(self, a_study, newidx):
        """Move a Study to a specific index in the Study list."""
        idx = self.studies.index(a_study)
        self.studies.pop(idx)
        self.studies.insert(newidx, a_study)
        self.save()
        self.reload()

    def move_study_earlier(self, a_study):
        """Move a Study one spot earlier in the Study list."""
        idx = self.studies.index(a_study)
        try:
            self.move_study_to_idx(a_study, idx - 1)
        except IndexError:
            pass

    def move_study_later(self, a_study):
        """Move a Study one spot later in the Study list."""
        idx = self.studies.index(a_study)
        try:
            self.move_study_to_idx(a_study, idx + 1)
        except IndexError:
            pass

    def move_study_first(self, a_study):
        """Move a Study to the front of the Study list."""
        self.move_study_to_idx(a_study, 0)

    def move_study_last(self, a_study):
        """Move a Study to the front of the Study list."""
        lastidx = len(self.studies)
        self.move_study_to_idx(a_study, lastidx)

    def rename(self, newname):
        """Rename the Stream."""
        self.name = newname
        self.save()
        self.reload()
        if self.active:
            scheduler.modify_job(str(self.id), name=self.name)

    def update_refresh(self, newrefresh):
        """Update the refresh interval."""
        self.refresh = newrefresh
        self.save()
        self.reload()
        if self.active:
            scheduler.reschedule_job(str(self.id), trigger='interval', seconds=self.refresh)

    def run(self):
        """Update the Study instances and collect alerts.

        Each Study will return alerts as a list of strings.
        """
        self.reload()
        logger.warning("%s: Updating %s", dt.now(tzlocal()), self.name)
        for study in self.studies:
            study.run()

    @classmethod
    def run_stream(cls, stream_id):
        """Static wrapper for `run`."""
        this_stream = cls.objects(id=stream_id).get()
        this_stream.run()

    def activate(self):
        """Periodically run the Stream."""
        scheduler.add_job(
            self.run_stream,
            args=[str(self.id)],
            trigger='interval',
            seconds=self.refresh,
            id=str(self.id),
            name=self.name,
            replace_existing=True,
        )

    def deactivate(self):
        """Stop the Stream from running periodically."""
        try:
            scheduler.remove_job(str(self.id))
        except JobLookupError:
            print("Job not found")

    def run_stream_once(self):
        """Submit a job to run the whole stream once."""
        scheduler.add_job(
            self.run_stream,
            args=[str(self.id)],
            id=str(self.id) + 'once',
            name=self.name + ' once',
            replace_existing=False,
        )

    @property
    def active(self):
        """Boolean value of whether the Stream is active in the scheduler."""
        return bool(scheduler.get_job(str(self.id)))

@util.handler(signals.pre_delete)
def clean_files(sender, document):  #pylint: disable=unused-argument
    """Signal deleted Studies to remove data files."""
    document.remove_files()

@clean_files.apply
class Study(Document):
    """Contains data process results."""
    # ID
    name = StringField(max_length=120, required=True)

    # Defining the analysis that generated the data
    process = ReferenceField('Process', required=True)
    valid_age = IntField(default=0)  # Zero means always valid

    # Data Inputs
    parents = MapField(ReferenceField('Study'))  # Precursor data used by the Analysis
    params = DictField()  # Processing parameters.

    # Data Outputs
    file = FileField()
    newfile = FileField()
    timestamp = DateTimeField(default=dt.now(tzlocal()))

    # Alerts
    alertslog = EmbeddedDocumentField('AlertsLog', default=AlertsLog())
    triggers = MapField(EmbeddedDocumentField('Trigger'))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    # pylint: disable=no-member
    # pylint: disable=unsupported-delete-operation
    # pylint: disable=not-a-mapping
    # pylint: disable=unsupported-assignment-operation
    def clean(self):
        """Clean attributes."""
        # Parents
        for key, parent in self.parents.items():
            if isinstance(parent, DBRef):
                del self.parents[key]
        # Alertslog
        self.alertslog.trim()

    ## Methods defining the Study ##

    def rename(self, newname):
        """Rename the Study."""
        self.name = newname
        self.save()
        self.reload()
        if self.active:
            scheduler.modify_job(str(self.id), name=self.name)

    def set_process(self, name):
        """Set the Study's Process based on a name."""
        self.process = Process.objects(name=name).get()
        self.save()

    def update_valid_age(self, new_age):
        """Update the valid age for the data."""
        self.valid_age = new_age
        self.save()
        self.reload()
        if self.active:
            scheduler.reschedule_job(str(self.id), trigger='interval', seconds=self.valid_age)

    ## Methods related to scheduling runs ##

    @property
    def valid(self):
        """Check if the Study data is still valid."""
        # Check the age of the data
        if self.valid_age == 0:
            current = True
        else:
            current = self.timestamp + dt.timedelta(seconds=self.valid_age) >= dt.now(tzlocal())
        # Check if the parents are valid too
        for parent in self.parents.values():
            if not parent.valid:
                current = False
                break
        return current

    @property
    def active(self):
        """Boolean value of whether the Study is active in the scheduler."""
        return bool(scheduler.get_job(str(self.id)))

    def activate(self):
        """Periodically run the Study."""
        scheduler.add_job(
            self.schedule_study,
            args=[str(self.id)],
            trigger='interval',
            seconds=self.valid_age,
            id=str(self.id),
            name=self.name,
            replace_existing=True,
        )

    def deactivate(self):
        """Stop the Study from running periodically."""
        try:
            scheduler.remove_job(str(self.id))
            scheduler.remove_job(str(self.id) + '_waiting')
        except JobLookupError:
            print("Job not found")

    def run_study_once(self):
        """Submit a job to run the whole Study once."""
        scheduler.add_job(
            self.schedule_study,
            args=[str(self.id)],
            id=str(self.id) + '_once',
            name=self.name + ' once',
            replace_existing=False,
        )

    @classmethod
    def schedule_study(cls, study_id):
        """Static wrapper for `schedule`."""
        this_study = cls.objects(id=study_id).get()
        this_study.schedule()

    def schedule(self):
        """Schedule the Study to run when all of its inputs are valid."""
        job_id = str(self.id) + '_waiting'
        if not bool(scheduler.get_job(job_id)):
            scheduler.add_job(
                self.queue_study,
                args=[str(self.id)],
                trigger='interval',
                seconds=5,
                id=job_id,
                name=self.name + ' waiting',
                replace_existing=True,
                max_instances=1,
            )

    @classmethod
    def queue_study(cls, study_id):
        """Static wrapper for `wait`."""
        this_study = cls.objects(id=study_id).get()
        this_study.queue()

    def queue(self):
        """Wait for the inputs to be valid and then run."""
        all_valid = True
        for parent in self.parents.values():
            if not parent.valid:
                all_valid = False
                parent.schedule()
        if all_valid:
            scheduler.remove_job(str(self.id) + '_waiting')
            self.run()

    def run(self):
        """Run the Study process on the inputs and return any alerts."""
        function = self.process.function
        parent_data = {name: study.data for name, study in self.parents.items()}
        self.data, newalerts = function(**parent_data, **self.params)
        self.timestamp = dt.now(tzlocal())
        self.alertslog.record_alerts(newalerts, self.timestamp)
        self.save()
        self.fire_alerts()

    ## Methods for handling inputs ##

    def add_parents(self, newparents):
        """Add all of the parents in the given dict of ids."""
        parent_objects = {key: Study.objects(id=val).get() for key, val in newparents.items()}
        self.parents.update(parent_objects)
        self.save()

    def add_params(self, newparams):
        """Add all of the params in the given dict."""
        self.params.update(newparams)
        self.save()

    def remove_inputs(self, inputs):
        """Remove all of the inputs in the given iterable of names."""
        for key in inputs:
            self.parents.pop(key, None)
            self.params.pop(key, None)
        self.save()

    @property
    def all_parents(self):
        """Full dict of parent kwargs, even if not set yet."""
        return {key: self.parents.get(key) for key in self.process.parents}

    @property
    def all_params(self):
        """Full dict of param kwargs, even if not set yet."""
        return {key: self.params.get(key) for key in self.process.params}


    ## Methods for handling the saved data ##

    @property
    def data(self):
        """Preprocess the data field to return the data in a usable format."""
        self.transfer_file()
        try:
            return pickle.loads(self.file.read())
        except TypeError:
            return None

    @data.setter
    def data(self, newdata):
        """Process the data for storage."""
        if newdata is None:
            self.remove_files()
        else:
            if not self.file:
                self.write_to(self.file, newdata)
            else:
                self.write_to(self.newfile, newdata)
                self.transfer_file()

    def write_to(self, field, newdata):
        """Write data to a FileField."""
        field.new_file()
        field.write(pickle.dumps(newdata))
        field.close()
        self.save()

    def transfer_file(self):
        """Transfer the data from newfile to file."""
        if self.newfile:
            newfile = self.newfile.read()
            self.file.replace(newfile)
            self.save()
            self.newfile.delete()
            self.save()

    def remove_files(self):
        """Remove the data."""
        self.file.delete()
        self.newfile.delete()
        self.save(validate=False)

    ## Methods for handling alerts ##

    @property
    def alerts(self):
        """Most recent alerts."""
        return self.alertslog.get_alerts(0)

    def clear_log(self):
        """Remove log entries."""
        self.alertslog.clear()
        self.save()

    def fire_alerts(self):
        """Fire alert triggers based on newly active and inactive alerts."""
        for trigger in self.triggers.values():
            trigger.check_conds(self)

    def get_trigger(self, trig_id):
        """Return the desired trigger."""
        return self.triggers.get(trig_id)

    def add_trigger(self, matchtext, **kwargs):
        """Add a Trigger to the Study, or update a matching one."""
        new = Trigger(matchtext=matchtext, **kwargs)
        self.triggers[str(new)] = new
        self.save()

    def del_trigger(self, trig_id):
        """Delete the specified trigger."""
        try:
            del self.triggers[trig_id]
            self.save()
        except KeyError:
            print(f"Trigger '{trig_id}' not found.")

class Process(Document):
    """Handles for choosing the appropriate data-processing functions.
    Parent arguments and parameters are parsed from the function docstring.
    """
    # Identity
    name = StringField(max_length=120, required=True, primary_key=True)

    # Args
    parents = ListField(StringField())
    params = ListField(StringField())

    # Meta
    schema_version = IntField(default=1)

    def clean(self):
        """Ensure the function is encoded properly."""
        # Set the args
        self.parents, self.params = self.get_proc_params(self.function)

    @property
    def function(self):
        """Get the function corresponding to the Process name."""
        return processes.ALL[self.name]

    def get_proc_params(self, func):
        """Return the names for the parent data and parameter arguments."""
        parents = []
        params = []
        docstr = inspect.getdoc(func)
        for line in docstr.splitlines():
            words = re.findall(r"[\w']+", line)
            if line.startswith('::parents::'):
                parents.extend(words[1:])
            elif line.startswith('::params::'):
                params.extend(words[1:])
        return parents, params
