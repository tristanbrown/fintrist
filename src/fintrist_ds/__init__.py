"""
Import all of the process functions.
"""
import sys
import inspect as ins

from fintrist_ds.scrapers.base import *
from fintrist_ds.analysis import *

# The process registry
CATALOG = dict(ins.getmembers(sys.modules[__name__], ins.isfunction))
# TODO: switch format to be based on list(scrapers.base__all__) + list(analysis__all__)
