"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import os
from dotenv import load_dotenv

import fintrist

load_dotenv()

class Config(fintrist.Config):
    APPDATA = os.path.join(os.getenv('APPDATA'), 'Fintrist')
    SECRET_KEY = 'mysecretkey'
