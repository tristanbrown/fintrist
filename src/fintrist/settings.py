"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config():
    APIKEY = os.getenv('APIKEY')
    APP_HOST = os.getenv('COMPUTERNAME')
    DATABASE_NAME = os.getenv('DB_NAME')
    local = int(os.getenv('DB_LOCAL') or 0)
    if local:
        USERNAME = None
        PASSWORD = None
        DB_HOST = 'localhost'
    else:
        USERNAME = os.getenv('DB_USERNAME')
        PASSWORD = os.getenv('DB_PASSWORD')
        DB_HOST = os.getenv('DB_HOST')
    DB_PORT = int(os.getenv('DB_PORT') or 5000)
    NUM_WORKERS = int(os.getenv('NUM_WORKERS') or 4)
