"""
Database Management

usage: ./manage.py [function]
"""
import sys

from mongoengine.errors import NotUniqueError
from .dask import client

__all__ = ['clear', 'restart_workers', 'close_client']

def clear(doc):
    """Delete all documents in the database collection."""
    doc.drop_collection()
    print(f"Cleared the {doc.__name__} database.")

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
