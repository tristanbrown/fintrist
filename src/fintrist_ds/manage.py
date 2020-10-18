"""
Database Management

usage: ./manage.py [function]
"""
import sys

from mongoengine.errors import NotUniqueError
from .catalog import CATALOG
from .dask import client
from fintrist.models import Recipe
from fintrist.services import register_recipe

__all__ = ['register', 'clear', 'restart_workers', 'close_client']

def register():
    """Register all of the processes as recipes in the database."""
    for name, func in CATALOG.items():
        try:
            register_recipe(name, func)
        except NotUniqueError:
            print(f"'{name}' skipped: Already registered.")
        except Exception as ex:
            print(f"'{name}' skipped: ")
            print(ex)

def clear():
    """Delete all recipes in the database."""
    Recipe.drop_collection()
    print("Cleared the recipes database.")

def restart_workers():
    """Restart the Dask workers to refresh the package cache."""
    client.restart()
    return

def close_client():
    """End the dask client."""
    client.close()
    return

def main():
    """Run the manage.py functions from the command line."""
    arg = sys.argv[1]
    globals()[arg]()

if __name__ == "__main__":
    main()
