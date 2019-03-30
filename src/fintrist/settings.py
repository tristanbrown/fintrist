"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import os
from dotenv import load_dotenv

load_dotenv()

APIKEY = os.getenv('APIKEY')
DATABASE_NAME = os.getenv('DATABASE_NAME')
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
