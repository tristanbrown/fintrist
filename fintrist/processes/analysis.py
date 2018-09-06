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
            'exists': self.any_data # Refactor to YAML file?
        }

        return analyses

    def any_data(self, inputs, study):
        """If there is data, raise alert."""
        if study.data.empty:
            study.alerts.add('Data Does Not Exist')
        else:
            study.alerts.add('Data Exists')
