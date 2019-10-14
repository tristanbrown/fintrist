"""Utility functions related to the web app."""

def get_choices(query):
    """Get a list of selection choices from a query object."""
    return [(str(item.id), item.name) for item in query]

def simplechoices(iterable):
    """Convert an iterable into a list of duplicated tuples."""
    return list(zip(iterable, iterable))
