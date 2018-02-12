"""
Stock tracking based on the Alpha Vantage api
https://github.com/RomelTorres/alpha_vantage
https://www.alphavantage.co/documentation/
https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/timeseries.py
"""

import matplotlib.pyplot as plt

class Chart():
    """
     
    """
    def __init__(self, series=None):
        self.series = series
        
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
    