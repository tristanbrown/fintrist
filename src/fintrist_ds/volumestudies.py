""""""
import arrow
import pandas as pd

from fintrist import get_study, create_study

class VolumeStudy():
    """Object for screening returns and characteristics of lists of
    stock symbols.

    Usage:
    symbols = ['SPY', 'TMO', 'MSFT', 'SONY', 'WD']
    screener = StockScreener("My Screening Set")
    data = screener.pull_stock_data(symbols)

    (to recover data later and process it):

    screener = StockScreener("My Screening Set")
    screener.gainloss_data()
    """

    def __init__(self, name="Volume Study"):
        self.name = name
        self.timestamp = arrow.now()

    ## Data I/O
    def pull_stock_data(self, symbols):
        """Pull the stock data from the source and save in fintrist."""
        from fintrist_ds import schedule_study  ## Dynamic import to avoid delays

        study = create_study(self.name, 'StockDaily', params={'symbol': symbols})
        schedule_study(study.name, force=True)
        return self.stock_data

    @property
    def stock_data(self):
        """Load the stock data saved in fintrist."""
        study = get_study(self.name)
        return study.data

    @property
    def symbols(self):
        return self.stock_data.index.get_level_values('symbol').unique()

    ## Volume Oscillator
