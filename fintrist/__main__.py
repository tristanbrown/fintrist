"""
Main routine of the package. 
"""

import sys
import os
import config
import time
from example_av import Example
from equity import Equity
from history import History
from sim import Ticker

def main(args=None):
    
    # Determine inputs.
    
    try:
        symbol = args[0]
    except IndexError:
        symbol = 'SPY'

    apikey = config.apikey
 
    
    # Examples
    # e = Example(apikey)
    # e.timeseries()
    # e.bbands()
    # e.sector()
    # e.crypto()
    # e.forex()

    # stock = Equity(apikey, symbol)
    # # print(stock.daily())
    # stock.refresh()
    # time0 = time.time()
    # path = os.path.join('data', symbol )
    # stock.daily(outputsize='full').to_csv(path_or_buf=path+'_daily.csv')
    # stock.sma().to_csv(path_or_buf=path+'_sma.csv')
    # stock.bbands().to_csv(path_or_buf=path+'_bbands.csv')
    # stock.macd().to_csv(path_or_buf=path+'_macd.csv')
    # stock.ultimate().to_csv(path_or_buf=path+'_ult.csv')

    lyb_daily = History('lyb', 'daily').data
    # print(lyb_daily)

    lyb_ult = History('lyb', 'ult').data
    # print(lyb_ult)

    lyb_sma = History('lyb', 'sma').data

    T = Ticker([lyb_daily, lyb_sma, lyb_ult])
    print(T.stream)

if __name__ == "__main__":
    main(sys.argv[1:])