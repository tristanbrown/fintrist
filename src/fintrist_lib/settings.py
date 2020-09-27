"""
This is the config file for fintrist_ds, containing various parameters the user may
wish to modify.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class ConfigObj():
    APIKEY_AV = os.getenv('APIKEY_AV')
    APIKEY_TIINGO = os.getenv('APIKEY_TIINGO')
    TZ = os.getenv('TIMEZONE') or 'UTC'

Config = ConfigObj()
