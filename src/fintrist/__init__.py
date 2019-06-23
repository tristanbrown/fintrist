"""
Start Fintrist, connect to the DB, and expose public methods.
"""

import mongoengine
from .settings import Config
from .models import *

__all__ = ('Study',)

mongoengine.connect(
    Config.DATABASE_NAME,
    host=Config.DB_HOST,
    port=Config.DB_PORT,
    username=Config.USERNAME,
    password=Config.PASSWORD,
    authentication_source='admin',
)
