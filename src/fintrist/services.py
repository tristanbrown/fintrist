"""Helper functions."""
import logging
from mongoengine.errors import SaveConditionError, DoesNotExist
from .models import Study, BaseStudy, Process, Recipe, Stream

logger = logging.getLogger(__name__)

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

def get_study(study_id):
    """Get a certain Study name or BaseStudy by name."""
    if isinstance(study_id, Study):
        return study_id
    elif isinstance(study_id, str):
        try:
            return BaseStudy.objects(name=study_id).get()
        except DoesNotExist:
            logger.debug(f"Study '{study_id}' does not exist.")

def get_data(name):
    """Get the data by a certain Study name or BaseStudy name."""
    obj = get_study(name)
    if obj:
        return obj.data

def get_process(name):
    """Get a certain Study name or BaseStudy by name."""
    try:
        return Process.objects(name=name).get()
    except DoesNotExist:
        logger.debug(f"Process '{name}' does not exist.")

def create_study(name, process, parents=None, params=None, **kwargs):
    """Use a local or library function to create a new Study."""
    new_procname = process.__name__
    existproc = get_process(new_procname)
    if existproc:
        newproc = existproc
    else:
        newproc = Process(name=new_procname, local=True)
        newproc.get_params(process)
        newproc.save()
    existstudy = get_study(name)
    if existstudy:
        newstudy = existstudy
        if kwargs:
            newstudy.update(**kwargs)
    else:
        newstudy = Study(name=name, process=newproc, **kwargs)
    if parents:
            newstudy.add_parents(parents)
    if params:
            newstudy.add_params(params)
    newstudy.save()
    return newstudy

def see_proc_args(name):
    proc = get_process(name)
    print(f"Parents: {proc.parents}")
    print(f"Params: {proc.params}")

def create_recipe(name, studyname, process, **kwargs):
    existproc = get_process(new_procname)
    existstudy = get_study(name)
    if existrecipe:
        newrecipe = existrecipe
        if kwargs:
            newrecipe.update(**kwargs)
    else:
        newrecipe = Recipe(name=name, studyname=studyname, process=existproc, **kwargs)
    newrecipe.get_metaparams()
    newrecipe.save()
    return newrecipe

def get_recipe(recipe_id):
    """Get a certain Recipe by name."""
    if isinstance(recipe_id, Recipe):
        return recipe_id
    elif isinstance(recipe_id, str):
        try:
            return Recipe.objects(name=recipe_id).get()
        except DoesNotExist:
            logger.debug(f"Recipe '{recipe_id}' does not exist.")

def create_stream(name, recipe_list):
    recipes = [get_recipe(recipe) for recipe in recipe_list]
    newstream = Stream(name=name, recipe_list=recipes)
    newstream.get_metaparams()
    newstream.save()
    return newstream
