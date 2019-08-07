"""
The engine that applies analyses to data and generates alerts.
"""
import numpy as np
import pandas as pd
import datetime as dt
import dateutil

__all__ = ['any_data', 'moving_avg', 'sample_dates',]

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
    alerts = []
    centering = False
    outdf = data[['close']]
    outdf = outdf.rename(columns={'close': 'close_price'})
    outdf['5-day avg'] = outdf.rolling(5, center=centering).mean()
    outdf['30-day avg'] = outdf['close_price'].rolling(30, center=centering).mean()
    outdf['100-day avg'] = outdf['close_price'].rolling(100, center=centering).mean()
    today = outdf.tail(1)
    diff_5_30 = today['5-day avg'] - today['30-day avg']
    if diff_5_30[0] > 0:
        alerts.append('MVA: 5d over 30d')
    else:
        alerts.append('MVA: 5d under 30d')
    return (outdf, alerts)

def sample_dates(data, N=100, window=365, backdate=0):
    """Sample the available dates in the data.
    ::parents:: data
    ::params:: N, window, backdate
    """
    try:
        backdt = dt.datetime.now() - dt.timedelta(days=int(backdate))
    except ValueError:
        backdt = dateutil.parser.parse(backdate)
    interval = data.index[
        (data.index > backdt - dt.timedelta(days=int(window))) & (data.index <= backdt)]
    sample_idx = np.random.choice(interval, int(N), replace=False)
    sample_idx.sort()
    alerts = ['complete']
    return (sample_idx, alerts)
