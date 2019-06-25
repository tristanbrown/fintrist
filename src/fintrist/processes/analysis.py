"""
The engine that applies analyses to data and generates alerts.
"""
import pandas as pd

__all__ = ['any_data', 'moving_avg']

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

def moving_avg(data):
    """Calculate the moving average values from the closing price.
    ::parents:: data
    """
    centering = False
    outdf = data[['close']]
    outdf = outdf.rename(columns={'close': 'close_price'})
    outdf['5-day avg'] = outdf.rolling(5, center=centering).mean()
    outdf['30-day avg'] = outdf['close_price'].rolling(30, center=centering).mean()
    outdf['100-day avg'] = outdf['close_price'].rolling(100, center=centering).mean()
    alerts = ['complete']
    return (outdf, alerts)
