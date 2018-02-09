"""
Stock tracking based on the Alpha Vantage api
https://github.com/RomelTorres/alpha_vantage
https://www.alphavantage.co/documentation/
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

    def quote(self):
        """Gives a real-time price quote for this equity."""
        data = self.intraday(outputsize='compact')[-1:]
        return data[['4. close','5. volume']]
    
    def intraday(self, interval='1min', outputsize='full'):
        """Gives the equity's Open, High, Low, Close, Volume, and Adj Close for
        each trading day, starting at the given date."""
        data, meta_data = self.ts.get_intraday(
            symbol=self.symbol,
            interval=interval,
            outputsize=outputsize
            )
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