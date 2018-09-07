"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import os

APIKEY = '***REMOVED***'
REFRESH_INTERVAL = 15
KEEP_STUDIES = 5
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATABASE_NAME = 'Fintrist_DB'
