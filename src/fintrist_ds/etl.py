"""ETL Processes"""

import numpy as np

__all__ = ['prep_pricing_data']

def prep_pricing_data(prices):
    data = prices.copy().drop(['close', 'high', 'low', 'open', 'volume'], axis=1)
    data = append_simquote(data)
    data = append_divyield(data)
    data = build_lookbacks(data)
    data = build_lookahead(data)
    return data.drop(['adjHigh', 'adjLow', 'adjClose', 'adjOpen', 'adjVolume', 'divCash'], axis=1)

def append_simquote(data):
    data['quote'] = data['adjLow'] + np.random.rand(len(data)) * (data['adjHigh'] - data['adjLow'])
    return data

def append_divyield(data):
    data['divyield'] = data['divCash'] / data['adjClose'].shift(1)

def append_cumulative(data, lookback):
    ref = data['adjClose'].shift(lookback)
    data[f'% cumul-{lookback}'] = (data['quote'] - ref)/ref
    return data

def append_pct_overnight(data, lookback):
    ref = data['adjClose'].shift(lookback + 1)
    data[f'% overnight-{lookback}'] = (data['adjOpen'].shift(lookback) - ref)/ref
    return data
    
def append_pct_day(data, lookback):
    ref = data['adjOpen'].shift(lookback)
    data[f'% day-{lookback}'] = (data['adjClose'].shift(1) - ref)/ref
    return data

def append_cum_vol_chg(data, lookback):
    ref = data['adjVolume'].shift(lookback + 1)
    data[f'% vol cumul-{lookback}'] = (data['adjVolume'].shift(1) - ref)/ref
    return data

def build_lookbacks(data):
    data = data.copy()
    recent_lookbacks = [1, 2, 3, 4, 5]
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

def build_lookahead(data, lookahead=30):
    data = data.copy()
    data['days to gain'] = np.NaN
    for i in range(1, lookahead+1): # 1 to 30
        check_future_gain(data, i)
        positive_today = data.loc[data['future gain'], 'future gain']
        data['days to gain'] = data['days to gain'].fillna(positive_today * i)
    data = data.drop('future gain', axis=1)
    return data