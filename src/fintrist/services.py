"""Helper functions."""
import logging
from mongoengine.errors import SaveConditionError, DoesNotExist

from . import migrations
from .models import Study, BaseStudy, Stream, Strategy
from fintrist_lib import get_recipe

logger = logging.getLogger(__name__)

def migrate():
    """Run all migrations."""
    migrations.upgrade()

def downgrade():
    """Run all downgrade migrations."""
    migrations.downgrade()

def store_data(data, name, overwrite=False):
    """Create a new BaseStudy or overwrite an existing one with the given data."""
    existing = get_study(name)
    if existing and overwrite:
        archive = existing
    elif existing:
        raise SaveConditionError(f"Study '{name}' already exists, but 'overwrite' set to False.")
    else:
        archive = BaseStudy(name=name)
    archive.data = data
    archive.save()
    return get_data(name)

def get_object(obj_id, obj_cls):
    """Get an object by name"""
    if isinstance(obj_id, obj_cls):
        return obj_id
    elif isinstance(obj_id, str):
        try:
            return obj_cls.objects(name=obj_id).get()
        except DoesNotExist:
            logger.debug(f"{obj_cls.__name__} '{obj_id}' does not exist.")

def get_study(study_id):
    """Get a certain Study name or BaseStudy by name."""
    return get_object(study_id, BaseStudy)

def get_data(name):
    """Get the data by a certain Study name or BaseStudy name."""
    obj = get_study(name)
    if obj:
        return obj.data

def create_study(name, recipe, parents=None, **kwargs):
    """Use a local or library function to create a new Study.

    recipe: str, function, Recipe
    """
    try:
        procname = recipe.__name__
    except AttributeError:
        procname = recipe
    existstudy = get_study(name)
    if existstudy:
        newstudy = existstudy
        kwargs['recipe'] = procname
        newstudy.update(**kwargs)
    else:
        newstudy = Study(name=name, recipe=procname, **kwargs)
    if parents:
            newstudy.set_parents(parents)
    newstudy.save()
    return newstudy

def create_stream(name, recipe_list):
    existstream = get_stream(name)
    if existstream:
        newstream = existstream
        existstream.recipes = recipe_list
    else:
        newstream = Stream(name=name, recipes=recipe_list)
    newstream.get_metaparams()
    newstream.save()
    return newstream

def get_stream(stream_id):
    """Get a certain Recipe by name."""
    return get_object(stream_id, Stream)

def spawn_study(rec_name, **kwargs):
    """Spawn a study from a recipe."""
    template = get_recipe(rec_name)
    recipe = template(**kwargs)
    newstudy = create_study(
        name=recipe.studyname,
        recipe=recipe,
        parents=spawn_parents(recipe),
        params=kwargs,
        valid_type=recipe.valid_type,
    )
    newstudy.save()
    return newstudy

def spawn_parents(recipe):
    """Spawn the parent studies required for a recipe."""
    parents = {
        parent_key: spawn_study(parent_name, **recipe.parent_params[parent_key])
        for parent_key, parent_name in recipe.parents.items()}
    return parents

def spawn_stream(stream_name, **kwargs):
    stream_obj = get_stream(stream_name)
    newstudies = [
        spawn_study(recipe, **kwargs) for recipe in stream_obj.recipes
    ]
    return newstudies

def get_strategy(strategy_id):
    """Get a certain Recipe by name."""
    return get_object(strategy_id, Strategy)

def create_strategy(name, **kwargs):
    existstrat = get_strategy(name)
    if existstrat:
        newstrat = existstrat
        if kwargs:
            newstrat.update(**kwargs)
    else:
        newstrat = Strategy(name=name, **kwargs)
    logger.debug(newstrat.to_json())
    newstrat.save()
    return newstrat

def create_backtest(study_name, strategy_name, period='1y'):
    """Create a Study that can be used to backtest."""
    model = get_study(study_name)
    strategy = get_strategy(strategy_name)
    backtest = create_study(
        f"{study_name}, {strategy_name} backtest", 'backtest',
        parents={'model': model},
        params={'strategy': strategy, 'period': period},
        valid_type='always')
    return backtest

def create_sim(symbol, backtest_name, **kwargs):
    prices = get_study(f"{symbol} daily")
    backtest = get_study(backtest_name)
    sim = create_study(
        f"{backtest_name.rstrip(' backtest')} Sim", 'simulate',
        parents={'prices': prices, 'backtest': backtest},
        params=kwargs,
        valid_type='always')
    return sim
