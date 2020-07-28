"""
The engine that applies analyses to studies.
"""
import datetime
from dateutil.tz import tzlocal

from dask.distributed import Client
client = Client()

from fintrist import (get_study, get_process, Process, BaseStudy,
    Study, create_study)

from .catalog import CATALOG
from .settings import Config


def build_dag(root_study, force=False):
    """Get the directed acyclic graph for this Study."""
    dag = {}
    for key, deps in root_study.dependencies.items():
        study_obj = BaseStudy.objects(id=key).get()
        if force or key == str(root_study.id):
            run = study_obj.run
        else:
            run = study_obj.run_if
        proc_func = get_function(study_obj.process.name)
        dag[key] = (run, proc_func, deps)
    return dag

def get_function(name):
    return CATALOG[name]

def schedule_study(a_study, force=False):
    """Schedule the Study to run when all of its inputs are valid."""
    client.get(build_dag(a_study, force), str(a_study.id), num_workers=Config.NUM_WORKERS)

def store_result(name, process, parents=None, params=None):
    """Use a local or library function to create and run a new Study."""
    if isinstance(process, str):
        process = get_function(process)
    newstudy = create_study(name, process, parents, params)
    newstudy.run(function=process)
    newstudy.save()
    return newstudy
