"""
Stock tracking based on the Alpha Vantage api
https://github.com/RomelTorres/alpha_vantage
https://www.alphavantage.co/documentation/
https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/timeseries.py
"""

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances

class Equity():
    """
     
    """
    def __init__(self, apikey=None, symbol='SPY'):
        self.symbol = symbol
        self.ts = TimeSeries(key=apikey, output_format='pandas')
        self.tech = TechIndicators(key=apikey, output_format='pandas')

    def refresh(self):
        """Clears all time-sensitive data."""
        properties = ['_quote', '_days']
        for data in properties:
            try:
                delattr(self, data)
            except AttributeError:
                pass
    
    @property
    def quote(self):
        """Gives a real-time price quote for this equity.
        Formatted as the date, last price, and cumulative day's volume."""
        try:
            return self._quote
        except AttributeError:
            data = self.daily(outputsize='compact')[-1:]
            self._quote = data[['4. close','5. volume']]
            return self._quote
    
    @property
    def days(self):
        """
        """
        try:
            return self._days
        except AttributeError:
            self._days = self.daily()
            return self._days
    
    def intraday(self, interval='1min', outputsize='compact'):
        """Gives the equity's Open, High, Low, Close, Volume, and Adj Close for
        each trading minute."""
        data, meta_data = self.ts.get_intraday(
            symbol=self.symbol,
            interval=interval,
            outputsize=outputsize,
            )
        return data

    def daily(self, outputsize='full'):
        """Gives the equity's Open, High, Low, Close, Volume, and Adj Close for
        each trading day, starting at the given date."""
        data, meta_data = self.ts.get_daily(
            symbol=self.symbol,
            outputsize=outputsize,
            )
        return data
    
    def weekly(self):
        """Gives the equity's Open, High, Low, Close, Volume, and Adj Close for
        each trading day, starting at the given date."""
        data, meta_data = self.ts.get_weekly(self.symbol)
        return data
    
    def monthly(self):
        """Gives the equity's Open, High, Low, Close, Volume, and Adj Close for
        each trading day, starting at the given date."""
        data, meta_data = self.ts.get_monthly(self.symbol)
        return data
    
    # Technical Indicators

    # See: https://stackoverflow.com/questions/3136915/passing-all-arguments-of-a-function-to-another-function
    def indic(self, function, interval='daily', time_period=20, price='close'):
        """Simple moving average time series. 
        """
        inputs = {
            'symbol': self.symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': 'close',
        }
        return function(**inputs)

    def sma(self, interval='daily', time_period=20, price='close'):
        """Simple moving average time series. 
        """
        data, meta_data = self.tech.get_sma(
            self.symbol,
            interval = interval,
            time_period = time_period,
            series_type='close',
            )
        return data

    def macd(self, interval='daily', time_period=20, series_type='close'):
        """Simple moving average time series. 
        """
        data, meta_data = self.tech.get_macd(
            self.symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            )
        return data
    
    def bbands(self, interval='daily', time_period=20, series_type='close',
                   nbdevup=None, nbdevdn=None, matype=None):
        """Bollinger Bands.
        """
        data, meta_data = self.tech.get_bbands(
            self.symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype,
            )
        return data
    
    def ultimate(self, symbol, interval='daily', timeperiod1=None,
                   timeperiod2=None, timeperiod3=None):
        """Ultimate oscillator.
        """
        data, meta_data = self.tech.get_bbands(
            self.symbol,
            interval = interval,
            time_period1 = time_period1,
            time_period2 = time_period2,
            time_period3 = time_period3,
            )
        return data