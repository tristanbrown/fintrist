"""
Start Fintrist, connect to the DB, and expose public methods.
"""

import mongoengine
from . import settings
from .models import Stream

__all__ = ('Stream',)

mongoengine.connect(settings.DATABASE_NAME)
