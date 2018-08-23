"""
Stock tracking based on the Alpha Vantage api
https://github.com/RomelTorres/alpha_vantage
"""

import matplotlib.pyplot as plt
from pprint import pprint
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange

class Example():
    """
    """
    def __init__(self, apikey):
        self.key = apikey
        self.format = 'pandas'
        self.symbol = 'LYB'

    def timeseries(self):
        ts = TimeSeries(key=self.key, output_format=self.format)
        data, meta_data = ts.get_intraday(symbol=self.symbol,interval='1min', outputsize='full')
        # print(data)
        data['4. close'].plot()
        plt.title('Intraday Time Series for the {} stock (1 min)'.format(self.symbol))
        plt.show()

    def bbands(self):
        ti = TechIndicators(key=self.key, output_format=self.format)
        data, meta_data = ti.get_bbands(symbol=self.symbol, interval='60min', time_period=60)
        data.plot()
        plt.title('BBbands indicator for  {} stock (60 min)'.format(self.symbol))
        plt.show()

    def sector(self):
        sp = SectorPerformances(key=self.key, output_format=self.format)
        data, meta_data = sp.get_sector()
        data['Rank A: Real-Time Performance'].plot(kind='bar')
        plt.title('Real Time Performance (%) per Sector')
        plt.tight_layout()
        plt.grid()
        plt.show()

    def crypto(self):
        cc = CryptoCurrencies(key=self.key, output_format=self.format)
        data, meta_data = cc.get_digital_currency_intraday(symbol='BTC', market='CNY')
        data['1b. price (USD)'].plot()
        plt.tight_layout()
        plt.title('Intraday value for bitcoin (BTC)')
        plt.grid()
        plt.show()

    def forex(self):
        cc = ForeignExchange(key=self.key)
        # There is no metadata in this call
        data, _ = cc.get_currency_exchange_rate(from_currency='BTC',to_currency='USD')
        pprint(data)