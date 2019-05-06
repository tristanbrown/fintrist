"""
The engine that applies analyses to data and generates alerts.
"""
import logging
import pickle # TODO: Switch to bson for performance? Consider different pickle protocols
import time
import inspect
import hashlib
import re
from datetime import datetime as dt
from dateutil.tz import tzlocal

from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (
    BinaryField, DateTimeField, DictField, EmbeddedDocumentField,
    EmbeddedDocumentListField, IntField, FileField,
    ListField, MapField, ReferenceField, StringField,
)
from mongoengine.errors import DoesNotExist
from mongoengine import signals
from apscheduler.jobstores.base import JobLookupError
from bson.dbref import DBRef

from fintrist import processes, util
from fintrist.scheduling import scheduler

__all__ = ('Stream', 'Study', 'Process', 'AlertsBoard')

logger = logging.getLogger(__name__)

class AlertsBoard(EmbeddedDocument):
    """Collects alerts for a particular Stream."""
    timestamp = DateTimeField(default=dt.now(tzlocal()))
    active = DictField()
    schema_version = IntField(default=1)

    meta = {'strict': False}

class Stream(Document):
    """Contains the relationships between Studies, Analyses, and AlertsBoards.

    Usage: Stream(name, refresh)
    """
    # pylint: disable=no-member
    name = StringField(max_length=120, required=True, unique=True)
    refresh = IntField(min_value=15)
    studies = ListField(ReferenceField('Study'))
    alertslog = EmbeddedDocumentListField('AlertsBoard')
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
        """Update the Study instances and collect alerts."""
        self.reload()
        logger.warning("%s: Updating %s", dt.now(tzlocal()), self.name)
        newalerts = {study.name: set(study.run()) for study in self.studies}
        self.update_alerts(newalerts)

    @staticmethod
    def run_stream(stream_id):
        """Static wrapper for `run`."""
        this_stream = Stream.objects(id=stream_id).get()
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

    def update_alerts(self, newalerts):
        """Add new alerts and choose whether to notify the user."""
        if self.alertslog:
            prev_board = self.alertslog[-1]
        else:
            prev_board = AlertsBoard()
        new_board = AlertsBoard(active=newalerts)
        self.update(push__alertslog=new_board)

@util.handler(signals.pre_delete)
def clean_files(sender, document):
    """Signal deleted Studies to remove data files."""
    document.remove_files()

@clean_files.apply
class Study(Document):
    """Contains data process results."""
    # ID
    name = StringField(max_length=120, required=True)

    # Defining the analysis that generated the data
    process = ReferenceField('Process', required=True)
    parents = MapField(ReferenceField('Study'))  # Precursor data used by the Analysis
    params = DictField()  # Processing parameters.

    # Outputs
    file = FileField()
    newfile = FileField()
    alerts = ListField(StringField(max_length=120))
    timestamp = DateTimeField(default=dt.now(tzlocal()))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    def clean(self):
        """Before saving, ensure process is an object ref."""
        if isinstance(self.process, str):
            self._set_process(self.process)

    def run(self):
        """Run the Study process on the inputs and return any alerts."""
        function = self.process.function
        parent_data = {name: study.data for name, study in self.parents.items()}
        self.data, self.alerts = function(**parent_data, **self.params)
        self.timestamp = dt.now(tzlocal())
        self.save()
        return self.alerts

    def rename(self, newname):
        """Rename the Study."""
        self.name = newname
        self.save()

    def set_process(self, name):
        """Saving wrapper for set_process."""
        self._set_process(name)
        self.save()

    def _set_process(self, name):
        """Set a new Process to the Study."""
        proc = Process().get_newest(name)
        self.process = proc

    def update_process(self):
        """Update the associated Process to the latest version."""
        name = self.process.name
        self.set_process(name)

    @property
    def all_parents(self):
        """Full dict of parent kwargs, even if not set yet."""
        return {key: self.parents.get(key) for key in self.process.parents}

    @property
    def all_params(self):
        """Full dict of param kwargs, even if not set yet."""
        return {key: self.params.get(key) for key in self.process.params}

    def remove_inputs(self, inputs):
        """Remove all of the inputs in the given iterable of names."""
        for key in inputs:
            self.parents.pop(key, None)
            self.params.pop(key, None)
        self.save()

    def add_parents(self, newparents):
        """Add all of the parents in the given dict of ids."""
        parent_objects = {key: Study.objects(id=val).get() for key, val in newparents.items()}
        self.parents.update(parent_objects)
        self.save()

    def add_params(self, newparams):
        """Add all of the params in the given dict."""
        self.params.update(newparams)
        self.save()

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
        self.save()

class Process(Document):
    """Handles for choosing the appropriate data-processing functions.
    Parent arguments and parameters are parsed from the function docstring.
    """
    # Identity
    name = StringField(max_length=120, required=True)
    checksum = StringField(required=True, primary_key=True)
    version = IntField(required=True)

    # Args
    parents = ListField(StringField())
    params = ListField(StringField())

    # Meta
    schema_version = IntField(default=1)

    def clean(self):
        """Encode a function as a hash."""
        if not isinstance(self.checksum, str) or len(self.checksum) != 40:
            new_func = self.function
            # Check the hash
            funcstring = str.encode(inspect.getsource(new_func))
            self.checksum = hashlib.sha1(funcstring).hexdigest()

            # Check for previous versions
            recent = self.get_newest(self.name)
            if recent:
                self.version = recent.version + 1
            else:
                self.version = 1

            # Set the args
            self.parents, self.params = self.get_proc_params(new_func)

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

    def get_newest(self, name):
        """Return the most recent version of a process."""
        return Process.objects(name=name).order_by("-version").limit(-1).first()
