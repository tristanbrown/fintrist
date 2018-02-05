"""
Main routine of the package. 
"""

import sys
import config
from example_av import Example

def main(args=None):
    
    # Determine inputs.
    
    if args is None:
        pass
        
    apikey = config.apikey
 
    
    # Examples
    e = Example(apikey)
    e.timeseries()
    # e.bbands()
    e.sector()
    e.crypto()
    e.forex()

    
if __name__ == "__main__":
    main(sys.argv[1:])