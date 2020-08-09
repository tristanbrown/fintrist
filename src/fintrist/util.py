"""Utility functions"""
import re

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
