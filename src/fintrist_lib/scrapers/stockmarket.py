"""
The engine that chooses a scraper and returns data.
"""
import pandas as pd
import arrow
import pandas_datareader as pdr
import pandas_market_calendars as mcal
from fintrist_lib.settings import Config

__all__ = ['stock', 'market_day', 'market_schedule', 'market_open']

def stock(symbol, frequency='daily', source=None, mock=None):
    """Get a stock quote history.

    ::parents:: mock
    ::params:: symbol, frequency, source
    ::alerts:: source: AV, source: Tiingo, ex-dividend, split, reverse split
    """
    ## Get the data from whichever source
    if mock is not None:
        source = 'mock'
    elif not source:
        source = 'Tiingo'
    if source == 'AV':
        data = pdr.get_data_alphavantage(symbol, api_key=Config.APIKEY_AV, start='1900')
        data.index = pd.to_datetime(data.index)
    elif source == 'Tiingo':
        data = pdr.get_data_tiingo(symbol, api_key=Config.APIKEY_TIINGO, start='1900')
        data = data.droplevel('symbol')  # Multiple stock symbols are possible
        data.index = data.index.date
    elif source == 'mock':
        data = mock

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

def market_schedule(start, end, tz=None):
    if tz is None:
        tz = Config.TZ
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start.datetime, end_date=end.datetime)
    try:
        for col in schedule.columns:
            schedule[col] = schedule[col].dt.tz_convert(tz)
    except AttributeError:
        pass
    return schedule, nyse

def market_open(now=None):
    nyse = mcal.get_calendar('NYSE')
    if now is None:
        now = arrow.now('America/New_York')
    schedule = nyse.schedule(start_date=now.datetime, end_date=now.datetime)
    return nyse.open_at_time(schedule, now.datetime)  # Market currently open


def market_day(symbol, source=None, mock=None):
    """Get a stock quote.

    ::parents:: mock
    ::params:: symbol, source
    ::alerts:: source: AV, source: Tiingo, ex-dividend, split, reverse split
    """
    print("Not implemented")
