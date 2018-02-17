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

    def comparison(self, function, chunk):
        """Takes each column in the chunk as a variable and inputs it into the
        function. The result of the function is returned. 
        """
        # Use "Time-aware Rolling" for this
        return (function(chunk))