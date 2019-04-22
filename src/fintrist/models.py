"""
The engine that applies analyses to data and generates alerts.
"""
import logging
import pickle # TODO: Switch to bson for performance? Consider different pickle protocols
import time
import hashlib
from datetime import datetime as dt
from dateutil.tz import tzlocal

from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (
    BinaryField, DateTimeField, DictField, EmbeddedDocumentField,
    EmbeddedDocumentListField, IntField,
    ListField, MapField, ReferenceField, StringField,
)
from mongoengine.errors import DoesNotExist
from apscheduler.jobstores.base import JobLookupError

from fintrist import processes
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

    # Figure out how to include class methods here (e.g. "add_study", "activate()")
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

class Study(Document):
    """Contains data process results."""
    # ID
    name = StringField(max_length=120, required=True)

    # Defining the analysis that generated the data
    process = ReferenceField('Process', required=True)
    inputs = DictField()  # Processing parameters.
    parents = MapField(ReferenceField('Study'))  # Precursor data used by the Analysis

    # Outputs
    data = BinaryField() # TODO: FileField? GridFS is messy for updates?
    alerts = ListField(StringField(max_length=120))
    timestamp = DateTimeField(default=dt.now(tzlocal()))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    def run(self):
        """Run the Study process on the inputs and return any alerts."""
        function = processes.ALL[self.process.name]
        newdata, newalerts = function(self.inputs)
        self.update(set__data=pickle.dumps(newdata))
        self.update(set__alerts=newalerts)
        self.update(set__timestamp=dt.now(tzlocal()))
        self.reload()
        return self.alerts

class Process(Document):
    """Handles for choosing the appropriate data-processing functions.
    Parent arguments and parameters are parsed from the function docstring.
    """
    # Identity
    name = StringField(max_length=120, required=True)
    fingerprint = StringField(required=True, primary_key=True)
    version = IntField(required=True)

    # Args
    parents = ListField(StringField())
    params = ListField(StringField())

    # Meta
    schema_version = IntField(default=1)
