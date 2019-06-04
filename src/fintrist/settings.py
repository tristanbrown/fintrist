"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config():
    APIKEY = os.getenv('APIKEY')
    DATABASE_NAME = os.getenv('DB_NAME')
    USERNAME = os.getenv('DB_USERNAME')
    PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = int(os.getenv('DB_PORT'))
