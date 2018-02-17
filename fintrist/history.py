"""
A module to create history objects, which contain stored stock historical
stock data. 
"""

import os
import pandas as pd

class History():
    """
    inputs: 
        symbol: stock symbol
        data_type: description of the type of data series

    outputs:
        self.data: a single data timeseries
    """
    def __init__(self, symbol=None, data_type='daily'):
        # Construct the filepath
        self.dir = 'data' # sets the data storage directory
        self.symbol = symbol
        self.type = data_type # string
        self.filepath = self.construct_path()

        # read the data
        self.data = self.read_data(self.filepath) 
        self.timestamp = self.get_timestamp(self.data)

    def read_data(self, source):
        """
        """
        data = self.validate_history(pd.read_csv(source, index_col=0))
        return data
    
    def validate_history(self, data):
        """Checks if the data is structured appropriately, and raises errors
        otherwise. 
        """
        # Need to implement 'raise' error checking here.
        return data

    def construct_path(self):
        """Constructs a specifically formatted filename and path to store the
        historical data in the proper location. 
        """
        return os.path.join(
                self.dir,
                '{0}_{1}.csv'.format(self.symbol, self.type),
                )

    def store(self):
        """Saves the history data into a file.
        """
        self.data.to_csv(path_or_buf=self.filepath)

    def get_timestamp(self, data):
        """Takes a dataframe containing stock history and returns the timestamp
        from when it was acquired.
        """

        # Implement this
        return ''

