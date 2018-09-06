"""
The engine that chooses a scraper and returns data.
"""

from fintrist import settings
from fintrist.scrapers.equity import Equity

class Scraper():
    """Chooses the appropriate analysis."""

    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs

    def get(self):
        """Return a particular scraper."""
        return self.database(self.inputs)[self.name]

    def database(self, inputs):
        """Return the scraper database."""
        scrapers = {
            'stock': Equity(settings.APIKEY, inputs['symbol'])
        }

        return scrapers
