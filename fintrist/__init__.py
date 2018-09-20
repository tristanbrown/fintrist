"""
Start Fintrist, connect to the DB, and expose public methods.
"""

import mongoengine
from . import settings
from .models import Stream, Study

__all__ = ('Stream',)

mongoengine.connect(settings.DATABASE_NAME)
