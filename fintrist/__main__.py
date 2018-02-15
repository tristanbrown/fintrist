"""
Main routine of the package. 
"""

import sys
import os
import config
import time
from example_av import Example
from equity import Equity

def main(args=None):
    
    # Determine inputs.
    
    if args is None:
        symbol = 'SPY'
    else:
        symbol = args[0]
        
    apikey = config.apikey
 
    
    # Examples
    # e = Example(apikey)
    # e.timeseries()
    # e.bbands()
    # e.sector()
    # e.crypto()
    # e.forex()

    stock = Equity(apikey, symbol)
    # print(stock.daily())
    stock.refresh()
    time0 = time.time()
    path = os.path.join('data', symbol )
    stock.daily(outputsize='full').to_csv(path_or_buf=path+'_daily.csv')
    stock.sma().to_csv(path_or_buf=path+'_sma.csv')
    stock.bbands().to_csv(path_or_buf=path+'_bbands.csv')
    stock.macd().to_csv(path_or_buf=path+'_macd.csv')
    stock.ultimate().to_csv(path_or_buf=path+'_ult.csv')
    
if __name__ == "__main__":
    main(sys.argv[1:])