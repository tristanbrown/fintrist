"""Helper functions."""
import logging
from mongoengine.errors import SaveConditionError, DoesNotExist

from . import migrations
from .models import Study, BaseStudy, Recipe, Stream, Strategy


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

# def get_process(process_id):
#     """Get a certain Study name or BaseStudy by name."""
#     return get_object(process_id, Process)

def create_study(name, process, parents=None, **kwargs):
    """Use a local or library function to create a new Study."""
    try:
        procname = process.__name__
    except AttributeError:
        procname = process
    existproc = get_recipe(procname)
    if not existproc:
        existproc = register_recipe(procname, process)
    existstudy = get_study(name)
    if existstudy:
        newstudy = existstudy
        if kwargs:
            newstudy.update(**kwargs)
            newstudy.reload()
    else:
        newstudy = Study(name=name, process=existproc, **kwargs)
    if parents:
            newstudy.set_parents(parents)
    newstudy.save()
    return newstudy

def see_proc_args(name):
    proc = get_recipe(name)
    print(f"Parents: {list(proc.parents.keys())}")
    print(f"Params: {list(proc.params.keys())}")

def register_recipe(name, func):
    new_proc = Recipe(name=name, process=name)
    new_proc.get_params(func)
    new_proc.get_metaparams()
    new_proc.save(force_insert=True)
    print(f"Inserted '{name}'.")
    return new_proc

def create_recipe(name, process, **kwargs):
    existrecipe = get_recipe(name)
    if existrecipe:
        newrecipe = existrecipe
        if kwargs:
            newrecipe.update(**kwargs)
            newrecipe.reload()
    else:
        newrecipe = Recipe(name=name, process=process, **kwargs)
    newrecipe.get_metaparams()
    logger.debug(newrecipe.to_json())
    newrecipe.save()
    return newrecipe

def get_recipe(recipe_id):
    """Get a certain Recipe by name."""
    return get_object(recipe_id, Recipe)

def create_stream(name, recipe_list):
    recipes = [get_recipe(recipe) for recipe in recipe_list]
    existstream = get_stream(name)
    if existstream:
        newstream = existstream
        existstream.recipes = recipes
    else:
        newstream = Stream(name=name, recipes=recipes)
    newstream.get_metaparams()
    newstream.save()
    return newstream

def get_stream(stream_id):
    """Get a certain Recipe by name."""
    return get_object(stream_id, Stream)

def spawn_study(rec_name, **kwargs):
    """Spawn a study from a recipe."""
    recipe = get_recipe(rec_name)
    newstudy = create_study(
        name=recipe.studyname.format(**kwargs),
        process=recipe,
        parents={key: val.format(**kwargs) for key, val in recipe.parents.items()},
        params={key: val.format(**kwargs) for key, val in recipe.params.items()},
        valid_age=recipe.valid_age,
        valid_type=recipe.valid_type,
    )
    newstudy.save()
    return newstudy

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
            newstrat.reload()
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
