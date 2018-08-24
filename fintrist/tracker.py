"""
The Tracking Engine that monitors the wall clock and any alerts, triggering
events such as data refresh.
"""
import time

from fintrist.scrapers import equity
from fintrist.settings import APIKEY

class TrackingEngine():
    """Tracking engine that monitors and updates data acquired by scrapers.

    Each scraper must return a dataframe.

    :param source: identifies the scraper to use to gather the data.
    :type source: str
    :param inputs: contains any parameters necessary for the scraper to run.
    :type inputs: dict
    """
    def __init__(self, source, inputs):
        self.source = source
        sources = ['stock']
        if source == sources[0]:
            self.scraper = equity.Equity(APIKEY, inputs['symbol'])
        else:
            raise ValueError(
                "Choose an appropriate data-scraper. "
                "Options: '{}'".format("', '".join(sources))
            )
        self.update()

    def track(self, interval=15):
        """Periodically update the data and run any desired analyses."""
        while True:
            print("{0}: Updating {1}".format(time.asctime(), self.source))
            self.update()
            print(self.data)
            time.sleep(interval)
    
    def update(self):
        """Update all of the tracked data."""
        self.scraper.refresh()
        self.data = self.scraper.get_data()

