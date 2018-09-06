"""
Database Management
"""
import sys

from mongoengine import connect
from fintrist import settings

def register():
    """Register all of the processes in the database."""
    connect(settings.DATABASE_NAME)

def main():
    """Check the command."""
    arg = sys.argv[1]
    if arg == 'register':
        register()

if __name__ == "__main__":
    main()
