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
            Process(name).save()
            print("Inserted '{}'.".format(name))
        except NotUniqueError:
            print("'{}' skipped: Already registered.".format(name))

def clear():
    """Delete all processes in the database."""
    connect(settings.DATABASE_NAME)
    Process.objects().delete()
    print("Cleared the processes database.")

def main():
    """Run the function specified by the command line argument."""
    arg = sys.argv[1]
    globals()[arg]()

if __name__ == "__main__":
    main()
