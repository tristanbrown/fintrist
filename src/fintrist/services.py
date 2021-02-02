"""Helper functions."""
import logging
from mongoengine.errors import SaveConditionError, DoesNotExist
from mongoengine.connection import _get_db
import gridfs

from . import migrations
from .models import Study, BaseStudy, Stream, Strategy, NNModel
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

def get_study(study_id=None, recipe=None, **kwargs):
    """Get a certain Study name or BaseStudy by name."""
    if recipe:
        recipe_obj = get_recipe(recipe)
        study_id = recipe_obj(**kwargs).studyname
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

def spawn_study(recipe, **kwargs):
    """Spawn a study from a recipe.
    Automatically generates parents.
    """
    if isinstance(recipe, str):
        template = get_recipe(recipe)
        recipe = template(**kwargs)
    newstudy = create_study(
        name=recipe.studyname,
        recipe=recipe.__name__,
        parents=spawn_parents(recipe),
        params=recipe.params,
        valid_type=recipe.valid_type,
    )
    return newstudy

def spawn_parents(recipe):
    """Spawn the parent studies required for a recipe."""
    parents = {key: spawn_study(parent_rec)
        for key, parent_rec in recipe.parents.items() if parent_rec}
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
        f"{study_name}, {strategy_name} backtest",
        'backtest',
        parents={'model': model},
        params={'strategy': strategy, 'period': period},
        valid_type='always')
    return backtest

def create_sim(symbol, backtest_name, **kwargs):
    prices = get_study(f"{symbol} daily")
    backtest = get_study(backtest_name)
    sim = create_study(
        f"{backtest_name.rstrip(' backtest')} Sim",
        'simulate',
        parents={'prices': prices, 'backtest': backtest},
        params=kwargs,
        valid_type='always')
    return sim

def create_nn(name, dataset, target_col, **kwargs):
    existstudy = get_study(name)
    if existstudy:
        newstudy = existstudy
        kwargs['target_col'] = target_col
        newstudy.update(**kwargs)
    else:
        newstudy = NNModel(name=name, target_col=target_col, **kwargs)
    newstudy.set_parents({'dataset': dataset})
    newstudy.save()
    return newstudy

def get_all_fileids():
    """Searches for all FileField entries on BaseStudy docs.
    Checks the following fields on all BaseStudy objects:
    - file
    - newfile
    - archive
    """
    id_list = []
    for doc in BaseStudy.objects():
        allfiles = [doc.file]
        for filemap in [doc.archive, doc.newfile, doc.fileversions]:
            allfiles += list(filemap.values())
        id_list += [filefield._id for filefield in allfiles if filefield]
    return id_list

def get_gridfs():
    db = _get_db()
    fs = gridfs.GridFS(db)
    return fs
    
def drop_gridfs_orphans():
    fs = get_gridfs()
    orphans = [entry._id for entry in fs.find() if entry._id not in get_all_fileids()]
    for orphan in orphans:
        print(f"Deleting gridfs file {orphan}")
        fs.delete(orphan)
