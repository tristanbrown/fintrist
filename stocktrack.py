# Doc: https://pypi.python.org/pypi/yahoo-finance
# Doc: http://pandas-datareader.readthedocs.io/en/latest/
# Doc: http://pandas.pydata.org/pandas-docs/stable/timeseries.html
# pandas_datareader is best for historical data, because it doesn't require
#   specific dates.
# yahoo_finance is necessary for realtime data.

from yahoo_finance import Share
from pandas_datareader import data
from datetime import datetime
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

class Equity(Share):
    """
    Based on the Share class from yahoo_finance module. Uses pandas_datareader
    to more conveniently deal with historical data, and adds plotting and data
    processing for indicators. 
    """
    def __init__(self, symbol='SPY'):
        super(Share, self).__init__(symbol)
        self._table = 'quotes'
        self._key = 'symbol'
        self.refresh()
        self.symbol = symbol
        self.price = self.get_price()
    
    def quote(self):
        """Gives a real-time price quote for this equity."""
        self.refresh()
        self.price = self.get_price()
        self.time = self.get_trade_datetime()
        return [self.price, self.time]
    
    def data(self, start='1900-01-01'):
        """Gives the equity's Open, High, Low, Close, Volume, and Adj Close for
        each trading day, starting at the given date."""
        return data.DataReader(self.symbol, 'yahoo', start)
        
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
        
        

lyb = Equity('LYB')
print(lyb.quote())
# lyb.plotprice('2016')
# lyb.plotvolume('2016')
lyb.plotPV('2014')
#pprint(lyb.get_historical('2015-07-25', '2015-07-30'))