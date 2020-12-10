"""
MongoDB connection
"""
import os
import mongoengine
from mongoengine.connection import _get_db
from .settings import Config

TESTNAME = 'Fintrist_Test'

def connect_db():
    mongoengine.disconnect()
    db_name = os.environ.get('ALT_DB') or Config.DATABASE_NAME
    mongoclient = mongoengine.connect(
        db_name,
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        username=Config.USERNAME,
        password=Config.PASSWORD,
        authentication_source='admin',
    )
    return mongoclient

def test_db(test=True):
    """Toggle the test DB."""
    if test:
        os.environ['ALT_DB'] = TESTNAME
    else:
        os.environ.pop('ALT_DB', None)
    mongoclient = connect_db()
    print(f"Test DB: {test}")
    return mongoclient

def drop_test():
    """Drop the test DB."""
    mongoclient = _get_db().client
    mongoclient.drop_database(TESTNAME)
    print(f"{TESTNAME} dropped.")
