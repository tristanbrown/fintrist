"""
Database Management
"""
import sys

from mongoengine import connect
from mongoengine.errors import NotUniqueError
from fintrist import settings
from fintrist import processes
from fintrist.models import Process

def register():
    """Register all of the processes in the database."""
    connect(settings.DATABASE_NAME)
    process_list = processes.ALL.keys()
    for name in process_list:
        try:
            Process(name=name).save()
        except NotUniqueError:
            pass

def main():
    """Check the command."""
    arg = sys.argv[1]
    if arg == 'register':
        register()

if __name__ == "__main__":
    main()
