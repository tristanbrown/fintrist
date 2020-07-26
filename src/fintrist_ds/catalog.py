import sys
import inspect as ins

from .scrapers.base import *
from .analysis import *

# The process registry
CATALOG = dict(ins.getmembers(sys.modules[__name__], ins.isfunction))
# TODO: switch format to be based on list(scrapers.base__all__) + list(analysis__all__)
