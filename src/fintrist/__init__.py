"""
Start Fintrist, connect to the DB, and expose public methods.
"""
from .connect import connect_db, test_db, drop_test
from .settings import Config
from .services import *
from .models import *
from fintrist_lib import get_recipe

__all__ = ('Study','BaseStudy', 'get_study', 'Strategy', 'migrate',
    'connect_db', 'mongoclient', 'test_db')

mongoclient, db = connect_db()
