"""
Start Fintrist, connect to the DB, and expose public methods.
"""
import mongoengine
from .settings import Config
from .services import *
from .models import *
from fintrist_lib import get_recipe

__all__ = ('Study','BaseStudy', 'get_study', 'Strategy', 'migrate')

mongoengine.connect(
    Config.DATABASE_NAME,
    host=Config.DB_HOST,
    port=Config.DB_PORT,
    username=Config.USERNAME,
    password=Config.PASSWORD,
    authentication_source='admin',
)
