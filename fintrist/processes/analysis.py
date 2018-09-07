"""
The engine that applies analyses to data and generates alerts.
"""

__all__ = ['any_data']

def any_data(self, study, inputs):
    """If there is data, raise alert."""
    if study.data.empty:
        study.alerts.add('Data Does Not Exist')
    else:
        study.alerts.add('Data Exists')
