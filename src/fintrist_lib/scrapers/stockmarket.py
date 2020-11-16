"""
The engine that chooses a scraper and returns data.
"""
import pandas as pd
import arrow
import pandas_datareader as pdr
import pandas_market_calendars as mcal

from alpaca_management.connect import trade_api
from fintrist_lib.settings import Config
from fintrist_lib.base import RecipeBase

__all__ = ['StockDaily', 'stock_intraday', 'market_schedule', 'market_open']

class StockDaily(RecipeBase):

    parents = {'mock': None}
    valid_type = 'market'

    def __init__(self, symbol='SPY'):
        self.studyname = f"{symbol} Stock Daily"
        self.params = {'symbol': symbol}

    @staticmethod
    def process(symbol='SPY', source=None, mock=None, **kwargs):
        """Get a stock quote history.

        ::parents:: mock
        ::params:: symbol, source
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

def stock_intraday(symbols, day=None, tz=None, source=None, mock=None):
    """Get intraday stock data.

    ::parents:: mock
    ::params:: symbols, day, tz, source
    ::alerts:: source: Alpaca, source: mock
    """
    ## Choose the source
    if mock is not None:
        source = 'mock'
    elif not source:
        source = 'Alpaca'

    ## Pick the day
    latest_day = latest_market_day(day)
    open_time = latest_day[0].isoformat()
    close_time = latest_day[1].isoformat()
    if tz is None:
        tz = Config.TZ

    ## Get the data
    if source == 'Alpaca':
        data = trade_api.get_barset(
            symbols, timeframe='minute', start=open_time, end=close_time, limit=1000)
        dfs = {symbol: format_stockrecords(records, tz) for symbol, records in data.items()}
    elif source == 'mock':
        dfs = mock
    
    if isinstance(symbols, str) or len(symbols) == 1:
        dfs = dfs[symbols]

    ## Create alerts
    alerts = [f'source: {source}']

    return (dfs, alerts)

def format_stockrecords(records, tz):
    """Reformat stock tick records as a dataframe."""
    df = pd.DataFrame.from_records(records.__dict__['_raw'])
    df = df.rename({
        'o': 'open', 'c': 'close',
        'l': 'low', 'h': 'high',
        'v': 'volume', 't': 'timestamp'}, axis=1
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)
    df = df.set_index('timestamp')
    return df

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
    """Is the market open?"""
    nyse = mcal.get_calendar('NYSE')
    if now is None:
        now = arrow.now('America/New_York')
    schedule = nyse.schedule(start_date=now.datetime, end_date=now.datetime)
    return nyse.open_at_time(schedule, now.datetime)  # Market currently open

def latest_market_day(now=None):
    """Get the hours of the most recent time when the market was open."""
    tz = 'America/New_York'
    if now is None:
        now = arrow.now(tz)
    schedule, nyse = market_schedule(now.shift(days=-7), now, tz)
    last_day = schedule.iloc[-1]
    if now.datetime < last_day['market_open']:
        return schedule.iloc[-2]
    else:
        return last_day
