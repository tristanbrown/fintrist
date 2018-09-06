"""
The engine that applies analyses to data and generates alerts.
"""
from datetime import datetime as dt
from dateutil.tz import tzlocal
from mongoengine import Document
from mongoengine import (DateTimeField, DictField, FileField, IntField,
                         ListField, MapField, ReferenceField, StringField)

class Stream(Document):
    """Contains the relationships between Studies, Analyses, and AlertsBoards."""
    name = StringField(max_length=120, required=True)
    studies = ListField(ReferenceField('Study'))
    alerts = ReferenceField('AlertsBoard')

class Study(Document):
    """Contains data process results."""
    # ID
    name = StringField(max_length=120, required=True)

    # Defining the analysis that generated the data
    analysis = ReferenceField('Process', required=True)
    inputs = DictField()  # Processing parameters.
    parents = MapField(ReferenceField('Study'))  # Precursor data used by the Analysis

    # Outputs
    data = FileField()
    alerts = ListField(StringField(max_length=120))
    timestamp = DateTimeField(default=dt.now(tzlocal()))

class Process(Document):
    """Handles for choosing the appropriate data-processing functions."""
    name = StringField(max_length=120, required=True)

class AlertsBoard(Document):
    """Collects alerts for a particular Stream."""
    id = IntField(required=True)
    active = ListField(StringField(max_length=120))
    log = ListField(StringField(max_length=120))
    timestamp = DateTimeField(default=dt.now(tzlocal()))
