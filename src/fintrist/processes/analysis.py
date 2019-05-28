"""
The engine that applies analyses to data and generates alerts.
"""
import pandas as pd

__all__ = ['any_data']

def any_data(data):
    """If there is data, raise alert.
    ::parents:: data
    """
    alerts = []
    if isinstance(data, pd.DataFrame) and not data.empty:
        alerts.append('data exists')
    else:
        alerts.append('data does not exist')
    return (None, alerts)
