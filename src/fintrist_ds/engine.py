"""
The engine that applies analyses to studies.
"""
from .dask import client

from fintrist import (get_study, BaseStudy, Study, create_study)
from fintrist_lib import get_recipe

from .backtest import backtest
from .settings import Config

__all__ = ['build_dag', 'schedule_study', 'store_result', 'run_study']

def build_dag(root_study, force=False):
    """Get the directed acyclic graph for this Study."""
    dag = {}
    for key, deps in root_study.dependencies.items():
        ## Must tag ids to avoid counting them as dag dependencies.
        key_tag = f"{key}_tag"
        root_id = f"{str(root_study.id)}_tag"
        dag[key] = (run_study, key_tag, root_id, force, deps)
    return dag

def run_study(key, root_id=None, force=False, depends=None):
    """Function to be scheduled:
    Query the Study and run it (if no longer valid).
    """
    key = key.split("_")[0]
    study_obj = BaseStudy.objects(id=key).get()
    recipe_name = study_obj.recipe
    if recipe_name == 'backtest':
        proc_func = backtest
    else:
        proc_func = get_recipe(recipe_name).process
    if force or key == root_id or root_id is None:
        study_obj.run(proc_func, force)
    else:
        study_obj.run_if(proc_func)

def schedule_study(a_study, force=False):
    """Schedule the Study to run when all of its inputs are valid."""
    a_study = get_study(a_study)
    dag = build_dag(a_study, force)
    client.get(dag, str(a_study.id), num_workers=Config.NUM_WORKERS)

def store_result(name, process, parents=None, params=None, **kwargs):
    """Use a local or library function to create and run a new Study."""
    if isinstance(process, str):
        process = get_recipe(process)
        function = process.process
    else:
        function = process
    newstudy = create_study(name, process, parents, params=params, **kwargs)
    newstudy.run(function=function)
    return newstudy
