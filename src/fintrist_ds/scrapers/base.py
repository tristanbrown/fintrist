"""
The engine that chooses a scraper and returns data.
"""
import pandas as pd
import pandas_datareader as pdr
from fintrist_ds.settings import Config

__all__ = ['stock']

def stock(symbol, frequency='daily', source=None):
    """Get a stock quote history.

    ::params:: symbol, frequency, source
    ::alerts:: data source
    """
    if not source:
        source = 'Tiingo'
    if source == 'AV':
        data = pdr.get_data_alphavantage(symbol, api_key=Config.APIKEY_AV, start='1900')
        data.index = pd.to_datetime(data.index)
    elif source == 'Tiingo':
        data = pdr.get_data_tiingo(symbol, api_key=Config.APIKEY_TIINGO, start='1900')
        data = data.droplevel('symbol')  # Multiple stock symbols are possible
        data.index = data.index.date
    alerts = [f'data source: {source}']
    return (data, alerts)
