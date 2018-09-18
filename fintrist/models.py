"""
The engine that applies analyses to data and generates alerts.
"""
import logging
import pickle # TODO: Switch to bson for performance? Consider different pickle protocols
import time

from datetime import datetime as dt
from dateutil.tz import tzlocal
from mongoengine.document import Document
from mongoengine.fields import (
    BinaryField, DateTimeField, DictField, EmbeddedDocumentField, IntField,
    ListField, MapField, ReferenceField, StringField,
)
from mongoengine.errors import DoesNotExist

from fintrist import processes

__all__ = ('Stream', 'Study', 'Process', 'AlertsBoard')

logger = logging.getLogger(__name__)

class AlertsBoard(Document):
    """Collects alerts for a particular Stream."""
    active = ListField(StringField(max_length=120))
    log = ListField(StringField(max_length=120))
    timestamp = DateTimeField(default=dt.now(tzlocal()))

    def add_alerts(self, newalerts):
        """Add new alerts to the AlertsBoard and trigger notifications."""
        logger.warning("Not yet implemented. New alerts: %s", newalerts)

class Stream(Document):
    """Contains the relationships between Studies, Analyses, and AlertsBoards.

    Usage: Stream(name, refresh)
    """
    name = StringField(max_length=120, required=True, unique=True)
    refresh = IntField(min_value=15)
    studies = ListField(ReferenceField('Study'))
    alerts = EmbeddedDocumentField('AlertsBoard')

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
        self.update(push__studies=newstudy.id)

    def activate(self):
        """Periodically update the Study instances and collect alerts."""
        self.reload()
        if not self.alerts:
            self.update(set__alerts=AlertsBoard())
            self.reload()
        while True:
            logger.info("%s: Updating %s", dt.now(tzlocal()), self.name)
            newalerts = {study.id: study.run() for study in self.studies}
            self.alerts.add_alerts(newalerts)
            time.sleep(self.refresh)

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
    """Handles for choosing the appropriate data-processing functions."""
    name = StringField(max_length=120, required=True, unique=True)
