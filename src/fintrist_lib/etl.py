"""ETL Processes"""

import numpy as np
from scipy import stats
from .base import RecipeBase
from .scrapers import stockmarket
from .simulate import StockDaySim

__all__ = ['prep_pricing_data', 'UpDownIndicator']

class UpDownIndicator(RecipeBase):

    valid_type = 'market'

    def __init__(self, symbol='SPY', timeofday=0.75, lookahead=1, threshold=0):
        self.studyname = f"{symbol} UpDownIndicator"
        self.parents = {
            'daily_prices': stockmarket.StockDaily(symbol),
            'today_prices': stockmarket.StockIntraday(symbol)
            }
        self.params = {
            'timeofday': timeofday,
            'lookahead': lookahead,
            'threshold': threshold,
            }

    @staticmethod
    def process(daily_prices, today_prices, timeofday, lookahead, threshold):
        """Prepare stock data for a trend length indicator.

        ::parents:: daily_prices, today_prices
        ::params::
        ::alerts::
        """
        data, alerts = prep_pricing_data(daily_prices, today_prices, timeofday)
        data['up indicator'] = check_future_gain(data, lookahead, threshold)
        data = data[['% overnight-0', '% day-0', '% day-1', '% cumul-1', '% cumul-10', '% cumul-30',
                      '% vol cumul-1', '% vol cumul-10', '% vol cumul-30', 'up indicator']]
        return data, alerts

def prep_pricing_data(daily_prices, today_prices, timeofday):
    """Prepare pricing data for a stock.

    ::parents:: daily_prices, today_prices
    ::params::
    ::alerts::

    Column Definitions
    quote: A price selected in the middle of the day.
            (historical is randomly chosen between the day's high and low).
    % day-0: % change between opening and quote.
    % day-N: % change between opening and closing, N-days ago.
    % overnight-N: % change, N-days ago, between previous closing and current open.
    % cumul-N: % change between closing N-days ago and quote today.
    """
    data = daily_prices.copy().drop(['close', 'high', 'low', 'open', 'volume'], axis=1)
    data = append_simquote(data, timeofday)
    data = append_today(data, today_prices)
    data = append_divyield(data)
    data = build_lookbacks(data)
    alerts = []
    return data, alerts

def append_simquote(data, timeofday):
    """Simulate a stock quote at a given time of day.

    timeofday (float): Given as the fraction of the market day passed
        at the time the quote is obtained.

    SOLVED THE QUOTE SIMULATION PROBLEM:
    https://stats.stackexchange.com/a/510059/297889
    """
    sim = StockDaySim(data['adjOpen'], data['adjClose'], data['adjLow'], data['adjHigh'], len(data))
    data['quote'] = sim.sample(timeofday)
    return data

def append_today(data, today_prices, div=0, split=1):
    today = today_prices.index[-1].date()
    data.loc[today, 'adjOpen'] = today_prices.iloc[0]['open']
    data.loc[today, 'quote'] = today_prices.iloc[-1]['close']
    if data.loc[today, ['divCash', 'splitFactor']].isnull().all():
        data.loc[today, ['divCash', 'splitFactor']] = [div, split]
    return data

def append_divyield(data):
    data['divyield'] = data['divCash'] / data['adjClose'].shift(1)
    return data

def append_pct_overnight(data, lookback):
    ref = data['adjClose'].shift(lookback + 1)
    data[f'% overnight-{lookback}'] = (data['adjOpen'].shift(lookback) - ref)/ref
    return data
    
def append_pct_day(data, lookback):
    ref = data['adjOpen'].shift(lookback)
    if lookback == 0:
        day_end = data['quote']
    else:
        day_end = data['adjClose']
    data[f'% day-{lookback}'] = (day_end.shift(lookback) - ref)/ref
    return data

def append_cumulative(data, lookback):
    ref = data['adjClose'].shift(lookback)
    data[f'% cumul-{lookback}'] = (data['quote'] - ref)/ref
    return data

def append_cum_vol_chg(data, lookback):
    ref = data['adjVolume'].shift(lookback + 1)
    data[f'% vol cumul-{lookback}'] = (data['adjVolume'].shift(1) - ref)/ref
    return data

def build_lookbacks(data):
    data = data.copy()
    recent_lookbacks = [0, 1, 2, 3, 4, 5]
    cum_lookbacks = [1, 2, 3, 4, 5, 10, 15, 30, 60]
    for lookback in recent_lookbacks:
        data = append_pct_overnight(data, lookback)
        data = append_pct_day(data, lookback)
    for lookback in cum_lookbacks:
        data = append_cumulative(data, lookback)
        data = append_cum_vol_chg(data, lookback)
    return data

def append_future_open(data, lookahead):
    data[f'open+{lookahead}'] = data['adjOpen'].shift(-lookahead)
    return data

def append_future_pct(data, lookahead):
    data[f'% open+{lookahead}'] = (data['adjOpen'].shift(-lookahead) - data['quote'])/data['quote']
    return data

def check_future_gain(data, lookahead, threshold=0):
    """For each day, check 'lookahead' number of days in the future. If that
    day's open is 'threshold' % higher than the simulated current-time quote,
    give True; otherwise False (or NA if there is no data that far ahead).
    """
    gain = (data['adjOpen'].shift(-lookahead) - data['quote'])/data['quote']
    check = gain > threshold
    return check[~gain.isna()].astype('Int64')

def build_daystogain(data, lookahead=2000):
    """For each day, count how many days in the future until the day opens
    higher than the simulated current-time quote."""
    data = data.copy()
    data['days to gain'] = np.NaN
    for i in range(1, lookahead+1):
        data['future gain'] = check_future_gain(data, i)
        positive_today = data.loc[data['future gain'].fillna(False), 'future gain']
        data['days to gain'] = data['days to gain'].fillna(positive_today * i)
    data = data.drop('future gain', axis=1)
    return data
