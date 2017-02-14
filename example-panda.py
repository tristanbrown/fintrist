# Doc: http://pandas-datareader.readthedocs.io/en/latest/

from pandas_datareader import data
from datetime import datetime

aapl = data.DataReader('AAPL', 'yahoo', '1980-01-01')

print(aapl.tail())
print(aapl.head())

ibm = data.get_data_yahoo(symbols='IBM', start=datetime(2000, 1, 1), end=datetime(2012, 1, 1))
print(ibm['Adj Close'])
print(ibm['Adj Close'][0])

spy = data.DataReader('SPY', 'yahoo')
print(spy)