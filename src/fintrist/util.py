"""Utility functions"""

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
