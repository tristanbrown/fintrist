"""
Database Management

usage: ./manage.py [function]
"""
import sys

from mongoengine.errors import NotUniqueError
from .catalog import CATALOG
from .dask import client
from fintrist.models import Process

__all__ = ['register', 'clear', 'restart_workers']

def register():
    """Register all of the processes in the database."""
    for name, func in CATALOG.items():
        try:
            new_proc = Process(name=name)
            new_proc.get_params(func)
            new_proc.save(force_insert=True)
            print(f"Inserted '{name}'.")
        except NotUniqueError:
            print(f"'{name}' skipped: Already registered.")
        except Exception as ex:
            print(f"'{name}' skipped: ")
            print(ex)

def clear():
    """Delete all processes in the database."""
    Process.drop_collection()
    print("Cleared the processes database.")

def restart_workers():
    """Restart the Dask workers to refresh the package cache."""
    client.restart()
    return

def main():
    """Run the manage.py functions from the command line."""
    arg = sys.argv[1]
    globals()[arg]()

if __name__ == "__main__":
    main()
