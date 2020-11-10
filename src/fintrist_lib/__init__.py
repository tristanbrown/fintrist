import inspect as ins
import types

from .scrapers import stockmarket
from . import analysis, etl, base

## The process registry ##

def get_catalog(module):
    recipes = ins.getmembers(module)
    return {name: recipe for name, recipe in recipes if name in module.__all__}

SCRAPERS_CATALOG = get_catalog(stockmarket)
ETL_CATALOG = get_catalog(etl)
ANALYSIS_CATALOG = get_catalog(analysis)
CATALOG = {**SCRAPERS_CATALOG, **ANALYSIS_CATALOG, **ETL_CATALOG}

def get_recipe(name):
    recipe = CATALOG[name]
    if isinstance(recipe, types.FunctionType):
        process = recipe
        recipe = base.RecipeBase
        recipe.process = process
        recipe.__name__ = process.__name__
    return recipe
