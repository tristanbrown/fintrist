"""
A library of analytical methods.
"""
import numpy as np
import pandas as pd
import datetime as dt
import dateutil

__all__ = ['any_data', 'moving_avg', 'sample_dates', 'simulate',
    'multisim']

def any_data(data):
    """If there is data, raise alert.
    ::parents:: data
    ::alerts:: data exists, data does not exist
    """
    alerts = []
    if isinstance(data, pd.DataFrame) and not data.empty:
        alerts.append('data exists')
    else:
        alerts.append('data does not exist')
    return (None, alerts)

def moving_avg(prices):
    """Calculate the moving average values from the closing price.
    ::parents:: prices
    ::alerts:: 5d over 30d, 5d under 30d
    """
    alerts = []
    centering = False
    outdf = prices[['close']]
    outdf = outdf.rename(columns={'close': 'close_price'})
    outdf['5-day avg'] = outdf.rolling(5, center=centering).mean()
    outdf['30-day avg'] = outdf['close_price'].rolling(30, center=centering).mean()
    outdf['100-day avg'] = outdf['close_price'].rolling(100, center=centering).mean()
    today = outdf.tail(1)
    diff_5_30 = today['5-day avg'] - today['30-day avg']
    if diff_5_30[0] > 0:
        alerts.append('5d over 30d')
    else:
        alerts.append('5d under 30d')
    return (outdf, alerts)

def sample_dates(data, N=100, window=365, backdate=0):
    """Sample the available dates in the data.
    ::parents:: data
    ::params:: N, window, backdate
    ::alerts:: complete
    """
    try:
        backdt = data.index[-1] - dt.timedelta(days=int(backdate))
    except ValueError:
        backdt = dateutil.parser.parse(backdate)
    interval = data[backdt - dt.timedelta(days=int(window)):backdt]
    sample_idx = np.random.choice(interval.index, int(N), replace=False)
    sample_idx.sort()
    alerts = ['complete']
    return (sample_idx, alerts)

def multisim(backtest, cash=10000, weightstep=0.1, confidence=2, days=50, N=10):
    """Run the simulation across various sampled date intervals.

    ::parents:: backtest
    ::params:: cash, weightstep, confidence, days, N
    ::alerts:: complete
    """
    days = int(days)
    window = backtest.index[-1] - backtest.index[0] - dt.timedelta(days=days)
    start_dates, _ = sample_dates(backtest, N=N, window=window.days, backdate=days)
    cols = ['start date', 'end date', 'market return', '% return', 'alpha']
    results = []
    for start in start_dates:
        start = pd.Timestamp(start)
        end = start + dt.timedelta(days=days)
        trial, _ = simulate(backtest, cash, weightstep, confidence, start, days)
        endvals = trial.tail(1)[['market return', '% return', 'alpha']].values[0]
        results.append([start, end, *endvals])
    data = pd.DataFrame(results, columns=cols)
    alerts = ['complete']
    return data, alerts

def simulate(backtest, cash=10000, weightstep=0.5, confidence=2, start_date=None, days=0):
    """Calculate the portfolio value gains/losses over the backtest period.

    ::parents:: backtest
    ::params:: cash, weightstep, confidence, start_date, days
    ::alerts:: complete
    """
    cash = int(cash)
    weightstep = float(weightstep)
    confidence = int(confidence)
    days = int(days)
    # Set up sub-intervals
    if start_date and days:
        if isinstance(start_date, str):
            start_date = dateutil.parser.parse(start_date)
        end_date = start_date + dt.timedelta(days=days)
        backtest = backtest[start_date:end_date]

    # Run the portfolio
    history = []
    portfolio = Portfolio(cash, 0, 0)

    for _, row in backtest.iterrows():
        portfolio.price = row['price']
        net_action = _calc_net_action(row['signals'], confidence)
        portfolio.weight_trade(net_action * weightstep)
        history.append(portfolio.as_dict.copy())

    history_df = pd.DataFrame(history, backtest.index)
    record = history_df[['cash', 'shares', 'value', 'market return', '% return', 'alpha']]
    alerts = ['complete']
    return (record, alerts)

def _calc_net_action(actions, confidence):
    """Calculate the net effective trading action based on signals."""
    net_action = 0
    if 'buy' in actions:
        net_action += 1
    if 'sell' in actions:
        net_action -= 1
    if 'strong buy' in actions:
        net_action += confidence
    if 'strong sell' in actions:
        net_action -= confidence
    return net_action

class Portfolio():
    """A finance portfolio that records and updates asset values.

    value: Total value of the portfolio.
    cash: Cash available for trading.
    price: Current price of the asset of interest.
    shares: Shares held of the asset.


    mkt_ret: Total % change in the underlying asset.
    pct_ret: Total % change in the Portfolio value.
    """
    def __init__(self, cash, shares, price):
        self.cash = cash
        self.shares = shares
        self.start_price = self.price = price
        self.start_value = self.value

    @property
    def as_dict(self):
        """Dictionary representation of Portfolio values."""
        adict = self.__dict__.copy()
        adict['value'] = self.value
        adict['% return'] = self.pct_ret
        adict['market return'] = self.mkt_ret
        adict['alpha'] = self.pct_ret - self.mkt_ret
        del adict['start_value']
        del adict['start_price']
        del adict['_price']
        adict['price'] = self.price
        return adict

    @property
    def price(self):
        """Current price of the asset of interest."""
        return self._price

    @price.setter
    def price(self, val):
        """Find the price if it wasn't initialized properly."""
        if not self.start_price:
            self.start_price = val
        self._price = val

    @property
    def value(self):
        """Total value of the portfolio."""
        return self.cash + self.shares * self.price

    @property
    def pct_ret(self):
        """Total % change in the Portfolio value."""
        return (self.value / self.start_value - 1) * 100

    @property
    def mkt_ret(self):
        """Total % change in the underlying asset."""
        return (self.price / self.start_price - 1) * 100

    def trade(self, order):
        """Try to buy or sell asset shares.

        order: Desired change in the number of shares.

        Updates shares and cash without bankrupting the Portfolio.
        """
        if order < 0:
            newshares = max(0, self.shares + order)
            actual_trade = newshares - self.shares
        elif order > 0:
            actual_trade = min(self.cash, order * self.price) // self.price
        else:
            actual_trade = 0
        self.shares += actual_trade
        self.cash -= actual_trade * self.price

    def weight_trade(self, weight):
        """Try to buy or sell asset shares.

        weight: Fraction of the Portfolio value to try to trade.
        """
        offer = weight * self.value
        order = offer // self.price
        self.trade(order)
