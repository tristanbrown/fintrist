"""ETL Processes"""

import numpy as np
from .base import RecipeBase
from .scrapers import stockmarket

__all__ = ['prep_pricing_data', 'TrendLengthData']

class TrendLengthData(RecipeBase):

    valid_type = 'market'

    def __init__(self, symbol='SPY'):
        self.studyname = f"{symbol} Trend Length Data"
        self.parents = {
            'daily_prices': stockmarket.StockDaily(symbol),
            'today_prices': stockmarket.StockIntraday(symbol)
            }

    @staticmethod
    def process(daily_prices, today_prices):
        """Prepare stock data for a trend length indicator.

        ::parents:: daily_prices, today_prices
        ::params::
        ::alerts::
        """
        data, alerts = prep_pricing_data(daily_prices, today_prices)
        data = build_daystogain(data)
        data = data.drop(['quote', 'adjHigh', 'adjLow', 'adjClose', 'adjOpen', 'adjVolume', 'divCash'], axis=1)
        return data, alerts

def prep_pricing_data(daily_prices, today_prices):
    """Prepare pricing data for a stock.

    ::parents:: daily_prices, today_prices
    ::params::
    ::alerts::
    """
    data = daily_prices.copy().drop(['close', 'high', 'low', 'open', 'volume'], axis=1)
    data = append_simquote(data)
    data = append_today(data, today_prices)
    data = append_divyield(data)
    data = build_lookbacks(data)
    alerts = []
    return data, alerts

def append_simquote(data):
    data['quote'] = data['adjLow'] + np.random.rand(len(data)) * (data['adjHigh'] - data['adjLow'])
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
    if np.isnan(data['adjClose'][-1 - lookback]):
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
    print(data.shape)
    return data

def append_future_open(data, lookahead):
    data[f'open+{lookahead}'] = data['adjOpen'].shift(-lookahead)
    return data

def append_future_pct(data, lookahead):
    data[f'% open+{lookahead}'] = (data['adjOpen'].shift(-lookahead) - data['quote'])/data['quote']
    return data

def check_future_gain(data, lookahead):
    gain = (data['adjOpen'].shift(-lookahead) - data['quote'])/data['quote']
    data[f'future gain'] = gain > 0
    return data

def build_daystogain(data, lookahead=2000):
    data = data.copy()
    data['days to gain'] = np.NaN
    for i in range(1, lookahead+1):
        check_future_gain(data, i)
        positive_today = data.loc[data['future gain'], 'future gain']
        data['days to gain'] = data['days to gain'].fillna(positive_today * i)
    data = data.drop('future gain', axis=1)
    return data
