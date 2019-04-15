"""
The engine that chooses a scraper and returns data.
"""

from fintrist.settings import Config
from fintrist.processes.scrapers.equity import Equity

__all__ = ['stock']

def stock(inputs):
    """Get a stock quote history."""
    scraper = Equity(Config().APIKEY, inputs['symbol'])
    if inputs['frequency'] == 'min':
        data = scraper.intraday()
    elif inputs['frequency'] == 'daily':
        data = scraper.daily()
    elif inputs['frequency'] == 'weekly':
        data = scraper.weekly()
    elif inputs['frequency'] == 'monthly':
        data = scraper.monthly()
    alerts = ['got data']
    return (data, alerts)
