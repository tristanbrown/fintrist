"""
The engine that applies analyses to data and generates alerts.
"""
import os
import pickle

class Study():
    """Analyzes the data and generates alerts."""

    def __init__(self, name, savedir, data):
        self.name = name
        self.savedir = savedir
        self.alerts = set()
        self.data = data
        self.output = None

    def save(self):
        """Saves the Study object in the database."""
        path = os.path.join(self.savedir, self.name)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Not yet implemented.")
    
    def delete(self):
        """Removes the Study object from the database."""
        print("Not yet implemented.")
