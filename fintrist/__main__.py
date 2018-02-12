"""
Main routine of the package. 
"""

import sys
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
    stock.daily(outputsize='compact')
    time1 = time.time()
    print(stock.quote)
    time2 = time.time()
    print(stock.quote)
    time3 = time.time()
    stock.refresh()
    print(stock.quote)
    time4 = time.time()

    print(time1-time0)
    print(time2-time1)
    print(time3-time2)
    print(time4-time3)
    
if __name__ == "__main__":
    main(sys.argv[1:])