"""
Start Fintrist, connect to the DB, and expose public methods.
"""

from mongoengine import connect
from . import settings
from .models import Stream

connect(settings.DATABASE_NAME)
