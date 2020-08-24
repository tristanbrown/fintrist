"""
The engine that applies analyses to studies.
"""
import datetime
from dateutil.tz import tzlocal

from .dask import client

from fintrist import (get_study, get_process, Process, BaseStudy,
    Study, create_study)

from .catalog import CATALOG
from .settings import Config

__all__ = ['build_dag', 'schedule_study', 'store_result', 'get_function']

def build_dag(root_study, force=False):
    """Get the directed acyclic graph for this Study."""
    dag = {}
    for key, deps in root_study.dependencies.items():
        ## Must tag ids to avoid counting them as dag dependencies.
        key_tag = f"{key}_tag"
        root_id = f"{str(root_study.id)}_tag"
        dag[key] = (run_study, key_tag, root_id, force, deps)
    return dag

def run_study(key, root_id, force=False, depends=None):
    """Function to be scheduled:
    Query the Study and run it (if no longer valid).
    """
    key = key.split("_")[0]
    study_obj = BaseStudy.objects(id=key).get()
    proc_func = get_function(study_obj.process.name)
    if force or key == root_id:
        study_obj.run(proc_func, force)
    else:
        study_obj.run_if(proc_func)

def get_function(name):
    return CATALOG[name]

def schedule_study(a_study, force=False):
    """Schedule the Study to run when all of its inputs are valid."""
    a_study = get_study(a_study)
    dag = build_dag(a_study, force)
    client.get(dag, str(a_study.id), num_workers=Config.NUM_WORKERS)

def store_result(name, process, parents=None, params=None, **kwargs):
    """Use a local or library function to create and run a new Study."""
    if isinstance(process, str):
        process = get_function(process)
    newstudy = create_study(name, process, parents, params, **kwargs)
    newstudy.run(function=process)
    newstudy.save()
    return newstudy

def backtest_study(name, process):
    """"""
    ## Create Backtest object, with desired study as the parent.

    ## Run Backtest object to generate history of alerts and buy/sell signals.
    ### - Every row is a run of the parent Study.
    ### - Each run generates alerts.
    ### - Alerts trigger the Study's Triggers, giving buy/sell signals.

    ## Run simulate to generate portfolio value over a section of the backtest.
    ### - This is where we specify the size of buy/sell actions.

    ## Or use multisim to generate multiple time-slice samples over which to evaluate
    ## the portfolio.
