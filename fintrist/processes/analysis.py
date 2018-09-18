"""
The engine that applies analyses to data and generates alerts.
"""

__all__ = ['any_data']

def any_data(inputs):
    """If there is data, raise alert."""
    alerts = []
    data = inputs['data']
    if not data:
        alerts.append('Data Does Not Exist')
    else:
        alerts.append('Data Exists')
    return (data, alerts)
