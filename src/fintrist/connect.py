"""
MongoDB connection
"""
import os
import mongoengine
from .settings import Config

def connect_db():
    mongoengine.disconnect()
    if os.environ.get('test') == 'True':
        db_name = 'Fintrist_Test'
    else:
        db_name = Config.DATABASE_NAME
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
    os.environ['test'] = str(test)
    connect_db()
    print(f"Test DB: {test}")
