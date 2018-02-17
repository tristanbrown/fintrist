"""
A module containing the objects that will traverse several streams of timeseries
data and report on conditional comparisons. 
"""

import pandas as pd

class Ticker():
    """
    """
    def __init__(self, streams=[]):
        if streams == []:
            raise IndexError('No datastream given.')
        elif type(streams) is type:
            self.stream = streams
        else:
            self.stream = self.align(streams)

    def align(self, list_of_streams):
        """
        """
        # This needs to combine several timeseries of different types into
        # a single dataframe that can be traversed. 
        
        stream = pd.concat(list_of_streams, axis=1)
        return stream
