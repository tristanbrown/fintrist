"""
Database Management

usage: ./manage.py [function]
"""
import sys

from mongoengine.errors import NotUniqueError
from fintrist_ds import CATALOG
from fintrist import util, client
from fintrist.models import Process

def register():
    """Register all of the processes in the database."""
    for name in CATALOG:
        try:
            Process(name=name).save(force_insert=True)
            print("Inserted '{}'.".format(name))
        except NotUniqueError:
            print("'{}' skipped: Already registered.".format(name))

def clear():
    """Delete all processes in the database."""
    Process.drop_collection()
    print("Cleared the processes database.")

def restart_workers():
    """Restart the Dask workers to refresh the package cache."""
    client.restart()

@util.not_implemented
def test():
    print("Test function")

def main():
    """Run the function specified by the command line argument."""
    arg = sys.argv[1]
    globals()[arg]()

if __name__ == "__main__":
    main()
