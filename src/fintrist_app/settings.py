"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import os
from dotenv import load_dotenv

load_dotenv()

APPDATA = os.path.join(os.getenv('APPDATA'), 'Fintrist')
