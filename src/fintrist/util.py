"""Utility functions"""
import re
import pandas_market_calendars as mcal

from fintrist import Config

def not_implemented(function):
    """"""
    print("Not yet implemented: '{}'".format(function.__name__))
    return function

def handler(event):
    """Signal decorator to allow use of callback functions as class decorators."""

    def decorator(fn):
        def apply(cls):
            event.connect(fn, sender=cls)
            return cls

        fn.apply = apply
        return fn

    return decorator

def get_variables(astr):
    """Get the bracketed variable names from a string."""
    return re.findall(r"\{([A-Za-z0-9_]+)\}", astr)

def market_schedule(start, end):
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start.datetime, end_date=end.datetime)
    for col in schedule.columns:
        schedule[col] = schedule[col].dt.tz_convert(Config.TZ)
    return schedule, nyse
