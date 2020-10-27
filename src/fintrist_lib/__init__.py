import inspect as ins

from .scrapers import stockmarket
from . import analysis, etl

## The process registry ##

def get_catalog(module):
    recipes = ins.getmembers(module)
    return {name: recipe for name, recipe in recipes if name in module.__all__}

SCRAPERS_CATALOG = get_catalog(stockmarket)
ETL_CATALOG = get_catalog(etl)
ANALYSIS_CATALOG = get_catalog(analysis)
CATALOG = {**SCRAPERS_CATALOG, **ANALYSIS_CATALOG, **ETL_CATALOG}
