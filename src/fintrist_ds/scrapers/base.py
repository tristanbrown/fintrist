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
    ::alerts:: source: AV, source: Tiingo, ex-dividend, split, reverse split
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

    ## Create alerts
    alerts = [f'source: {source}']
    div = data.loc[data.index.max(), 'divCash']
    if div > 0:
        alerts.append('ex-dividend')
    splitf = data.loc[data.index.max(), 'splitFactor']
    if splitf > 1:
        alerts.append('split')
    elif splitf < 1:
        alerts.append('reverse split')
    return (data, alerts)
