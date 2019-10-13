"""
The engine that chooses a scraper and returns data.
"""
import pandas as pd
from fintrist import Config
from fintrist_ds.scrapers.equity import Equity

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
    data.index = pd.to_datetime(data.index)
    data.columns = [col.split()[1] for col in data.columns]
    alerts = ['got data']
    return (data, alerts)
