"""
Stock tracking based on the Alpha Vantage api
https://github.com/RomelTorres/alpha_vantage
https://www.alphavantage.co/documentation/
https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/timeseries.py
"""

import matplotlib.pyplot as plt
from pprint import pprint
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances

class Equity():
    """
     
    """
    def __init__(self, apikey=None, symbol='SPY'):
        self.symbol = symbol
        self.ts = TimeSeries(key=apikey, output_format='pandas')

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
        data, meta_data = self.ts.get_weekly(symbol=self.symbol)
        return data
    
    def monthly(self):
        """Gives the equity's Open, High, Low, Close, Volume, and Adj Close for
        each trading day, starting at the given date."""
        data, meta_data = self.ts.get_monthly(symbol=self.symbol)
        return data
    
    def batch(self, symbols):
        """
        """
        data, meta_data = self.ts.get_batch_stock_quotes(symbols)
        return data
        
    def plot(self, series, start='1900-01-01'):
        """Creates a plot of a specific series for the equity, starting at the
        given date."""
        y = self.data(start)[series]
        x = y.index
        plt.plot(x, y, '-b')
        plt.ylabel(series)
        
    
    def plotprice(self, start='1900-01-01'):
        """Plots the equity's Adj Close price, starting at the given date."""
        self.plot('Adj Close', start)
        plt.xlabel("Date")
        plt.title(self.get_name())
        plt.show()
    
    def plotvolume(self, start='1900-01-01'):
        """Plots the equity's Volume, starting at the given date."""
        self.plot('Volume', start)
        plt.xlabel("Date")
        plt.title(self.get_name())
        plt.show()
    
    def plotPV(self, start='1900-01-01'):
        plt.subplot(2, 1, 1)
        self.plot('Adj Close', start)
        plt.title(self.get_name())
        
        plt.subplot(2, 1, 2)
        self.plot('Volume', start)
        plt.xlabel("Date")
        plt.show()
    
    # def SMA(self, start='1900-01-01', days):
        # prices = self.data(start)['Adj Close']