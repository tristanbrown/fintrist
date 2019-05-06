"""
The engine that chooses a scraper and returns data.
"""

from fintrist.settings import Config
from fintrist.processes.scrapers.equity import Equity

__all__ = ['stock']

def stock(symbol, frequency):
    """Get a stock quote history.

    ::params:: symbol, frequency
    """
    scraper = Equity(Config().APIKEY, symbol)
    if frequency == 'min':
        data = scraper.intraday()
    elif frequency == 'daily':
        data = scraper.daily()
    elif frequency == 'weekly':
        data = scraper.weekly()
    elif frequency == 'monthly':
        data = scraper.monthly()
    alerts = ['got data']
    return (data, alerts)
