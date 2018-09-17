"""
The engine that applies analyses to data and generates alerts.
"""
import logging

from datetime import datetime as dt
from dateutil.tz import tzlocal
from mongoengine import Document
from mongoengine import (DateTimeField, DictField, FileField, IntField,
                         ListField, MapField, ReferenceField, StringField)
from mongoengine.errors import DoesNotExist

__all__ = ('Stream', 'Study', 'Process', 'AlertsBoard')

logger = logging.getLogger(__name__)

class Stream(Document):
    """Contains the relationships between Studies, Analyses, and AlertsBoards.

    Usage: Stream(name, refresh)
    """
    name = StringField(max_length=120, required=True, unique=True)
    refresh = IntField(min_value=15)
    studies = ListField(ReferenceField('Study'))
    alerts = ReferenceField('AlertsBoard') # Turn this into EmbeddedDocument type?

    # Figure out how to include class methods here (e.g. "add_study", "activate()")
    def add_study(self, name, proc_name, inputs):
        """Create a new study tracked by a Stream instance."""
        try:
            self.reload()
        except DoesNotExist as ex:
            logger.error("Please save the Stream before trying to create a Study.")
            raise ex
        proc_ref = Process.objects(name=proc_name).get().id
        newstudy = Study(name, proc_ref, inputs)
        newstudy.save()
        self.update(push__studies=newstudy.id)

class Study(Document):
    """Contains data process results."""
    # ID
    name = StringField(max_length=120, required=True)

    # Defining the analysis that generated the data
    process = ReferenceField('Process', required=True)
    inputs = DictField()  # Processing parameters.
    parents = MapField(ReferenceField('Study'))  # Precursor data used by the Analysis

    # Outputs
    data = FileField()
    alerts = ListField(StringField(max_length=120))
    timestamp = DateTimeField(default=dt.now(tzlocal()))

class Process(Document):
    """Handles for choosing the appropriate data-processing functions."""
    name = StringField(max_length=120, required=True, unique=True)

class AlertsBoard(Document):
    """Collects alerts for a particular Stream."""
    active = ListField(StringField(max_length=120))
    log = ListField(StringField(max_length=120))
    timestamp = DateTimeField(default=dt.now(tzlocal()))
