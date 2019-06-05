"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Configuration options.")
parser.add_argument('--local', action='store_true')
args = parser.parse_args()

class Config():
    APIKEY = os.getenv('APIKEY')
    APP_HOST = os.getenv('COMPUTERNAME')
    DATABASE_NAME = os.getenv('DB_NAME')
    if args.local:
        USERNAME = None
        PASSWORD = None
        DB_HOST = 'localhost'
    else:
        USERNAME = os.getenv('DB_USERNAME')
        PASSWORD = os.getenv('DB_PASSWORD')
        DB_HOST = os.getenv('DB_HOST')
    DB_PORT = int(os.getenv('DB_PORT'))
