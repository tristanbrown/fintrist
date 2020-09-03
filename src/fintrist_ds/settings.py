"""
This is the config file for fintrist_ds, containing various parameters the user may
wish to modify.
"""
import os
from dotenv import load_dotenv

import fintrist

load_dotenv()

class ConfigObj():
    APIKEY_AV = os.getenv('APIKEY_AV')
    APIKEY_TIINGO = os.getenv('APIKEY_TIINGO')
    DASK_PORT = int(os.getenv('DASK_PORT') or 8786)
    NUM_WORKERS = int(os.getenv('NUM_WORKERS') or 4)
    DASK_HOST = f"{fintrist.Config.DB_HOST}:{DASK_PORT}"
    TZ = fintrist.Config.TZ

Config = ConfigObj()
