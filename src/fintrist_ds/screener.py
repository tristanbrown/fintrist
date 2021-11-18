""""""
import arrow
import pandas as pd

from fintrist import get_study, create_study

class StockScreener():
    """Object for screening returns and characteristics of lists of
    stock symbols.

    Usage:
    symbols = ['SPY', 'TMO', 'MSFT', 'SONY', 'WD']
    screener = StockScreener("My Screening Set")
    data = screener.pull_stock_data(symbols)

    (to recover data later and process it):

    screener = StockScreener("My Screening Set")
    screener.gainloss_data()
    """

    def __init__(self, name="Stock Screener"):
        self.name = name
        self.timestamp = arrow.now()

        self.crashes = {
            '2020Feb': ('2020-02-19', '2020-03-23'),
            '2018Oct': ('2018-10-03', '2018-12-24'),
            '2015Aug': ('2015-07-22', '2016-02-11'),
            '2011May': ('2011-05-02', '2011-10-03'),
            '2010Apr': ('2010-04-26', '2010-08-30'),
            '2007Oct': ('2007-10-09', '2009-03-09'),
        }
        self.gains = {
            '1y': years_interval(1),
            '3y': years_interval(3),
            '5y': years_interval(5),
            '10y': years_interval(10),
            '20y': years_interval(20),
        }
        self.time_periods = self.crashes.copy()
        self.time_periods.update(self.gains)

    ## Data I/O
    def pull_stock_data(self, symbols):
        """Pull the stock data from the source and save in fintrist."""
        from fintrist_ds import schedule_study  ## Dynamic import to avoid delays

        study = create_study(self.name, 'StockDaily', params={'symbol': symbols})
        schedule_study(study.name, force=True)
        return self.stock_data

    @property
    def stock_data(self):
        """Load the stock data saved in fintrist."""
        study = get_study(self.name)
        return study.data

    @property
    def symbols(self):
        return self.stock_data.index.get_level_values('symbol').unique()

    ## Display Gains and Losses 
    def gainloss_data(self):
        """Return table of gains and losses"""
        data = self.stock_data
        all_symbols = self.symbols
        result = [self.check_all_intervals(data.loc[symbol,:]) for symbol in all_symbols]
        df = pd.DataFrame(result, index=all_symbols, columns = self.time_periods.keys())
        df['crash_mean'] = df[self.crashes.keys()].mean(axis=1)
        self.append_annual(df)
        return df.sort_values('crash_mean', ascending=False)

    def check_all_intervals(self, prices):
        """Calculate returns across every given time interval."""
        return [price_returns(prices, *interval) for interval in self.time_periods.values()]

    def append_annual(self, returns_df):
        """Append annual returns"""
        for period in self.gains.keys():
            years = int(period.rstrip('y'))
            annualized = annual_from_long(returns_df[period], years)
            returns_df[f"{period}_ann"] = annualized

def get_date(datestr):
    """Get a date from a date string."""
    return arrow.get(datestr).date()

def years_interval(years, end=None):
    """Specify the number of years back and get the dates."""
    if end is None:
        end = arrow.now()
    start = end.shift(years=-years)
    return (str(start.date()), str(end.date()))

def get_last_price(prices, lastdate):
    """Get the last available price at a given date."""
    last_prices = prices.loc[:arrow.get(lastdate).date()]
    try:
        return last_prices.iloc[-1]['adjClose']
    except IndexError:
        return

def price_returns(prices, start, end):
    """Calculate the returns on a time interval."""
    start_price = get_last_price(prices, start)
    end_price = get_last_price(prices, end)
    try:
        return end_price / start_price - 1
    except TypeError:
        return

def annual_from_long(long_return, years):
    """Calculate the annual returns from long-term returns."""
    return (1 + long_return) ** (1 / years) - 1
