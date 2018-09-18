"""
Import all of the process functions.
"""
import sys
import inspect as ins

from fintrist.processes.scrapers.base import *
from fintrist.processes.analysis import *

# The process registry
ALL = dict(ins.getmembers(sys.modules[__name__], ins.isfunction))
# TODO: switch format to be based on list(scrapers.base__all__) + list(analysis__all__)
