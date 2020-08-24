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

def get_process(process_id):
    """Get a certain Study name or BaseStudy by name."""
    return get_object(process_id, Process)

def create_study(name, process, parents=None, params=None, **kwargs):
    """Use a local or library function to create a new Study."""
    existproc = get_process(process)
    if existproc:
        newproc = existproc
    else:
        newproc = Process(name=process.__name__, local=True)
        newproc.get_params(process)
        newproc.save()
    existstudy = get_study(name)
    if existstudy:
        newstudy = existstudy
        if kwargs:
            newstudy.update(**kwargs)
            newstudy.reload()
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
    existproc = get_process(process)
    existrecipe = get_recipe(name)
    if existrecipe:
        newrecipe = existrecipe
        if kwargs:
            newrecipe.update(**kwargs)
            newrecipe.reload()
    else:
        newrecipe = Recipe(name=name, studyname=studyname, process=existproc, **kwargs)
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
        process=recipe.process,
        parents={key: val.format(**kwargs) for key, val in recipe.parents.items()},
        params={key: val.format(**kwargs) for key, val in recipe.params.items()},
        valid_age=recipe.valid_age,
        valid_type=recipe.valid_type,
    )
    for trigger in recipe.triggers.values():
        newstudy.add_trigger(
            trigger.matchtext.format(**kwargs),
            on=trigger.on,
            actions=trigger.actions
        )
    newstudy.save()
    return newstudy

def spawn_stream(stream_name, **kwargs):
    stream_obj = get_stream(stream_name)
    newstudies = [
        spawn_study(recipe, **kwargs) for recipe in stream_obj.recipes
    ]
    return newstudies
