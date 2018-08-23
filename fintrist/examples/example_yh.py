# Doc: https://pypi.python.org/pypi/yahoo-finance

from yahoo_finance import Share
from pprint import pprint

yahoo = Share('YHOO')
print(yahoo.get_open())
print(yahoo.get_price())
print(yahoo.get_trade_datetime())

yahoo.refresh()
print(yahoo.get_price())
print(yahoo.get_trade_datetime())

pprint(yahoo.get_historical('2015-07-25', '2015-07-30'))

pprint(yahoo.get_info())