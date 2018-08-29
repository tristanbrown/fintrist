"""
The engine that applies analyses to data and generates alerts.
"""

class Analysis():
    """Chooses the appropriate analysis."""

    def __init__(self, name, inputs):
        database = self.database()
        self.analyze = lambda study: database[name](inputs, study)

    def database(self):
        """Return the analysis database."""
        analyses = {
            'exists': self.any_data
        }

        return analyses

    def any_data(self, inputs, study):
        """If there is data, raise alert."""
        if study.data.empty:
            study.alerts.add('Data Does Not Exist')
        else:
            study.alerts.add('Data Exists')

class Study():
    """Analyzes the data and generates alerts."""

    def __init__(self, data):
        self.alerts = set()
        self.data = data
        self.output = None

    def save(self):
        """Saves the Study object in the database."""
        print("Not yet implemented.")
    
    def delete(self):
        """Removes the Study object from the database."""
        print("Not yet implemented.")
