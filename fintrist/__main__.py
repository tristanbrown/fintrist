"""
Main routine of the package. 
"""

import sys
import config
from example_av import Example
from equity import Equity

def main(args=None):
    
    # Determine inputs.
    
    if args is None:
        pass
        
    apikey = config.apikey
 
    
    # Examples
    # e = Example(apikey)
    # e.timeseries()
    # e.bbands()
    # e.sector()
    # e.crypto()
    # e.forex()

    nvda = Equity(apikey, 'NVDA')
    print(nvda.intraday())
    print(nvda.quote())

    
if __name__ == "__main__":
    main(sys.argv[1:])