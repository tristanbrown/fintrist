"""
The engine that applies analyses to studies.
"""
import datetime
from dateutil.tz import tzlocal

from dask.distributed import Client
client = Client()

from fintrist import get_study, get_process, Process, BaseStudy, Study

from catalog import CATALOG
from settings import Config

def store_result(name, process, parents=None, params=None):
    """Use a local function to create a new Study."""
    new_procname = process.__name__
    existproc = get_process(new_procname)
    if existproc:
        newproc = existproc
    else:
        newproc = Process(name=new_procname, local=True)
        newproc.save()
    existstudy = get_study(name)
    if existstudy:
        newstudy = existstudy
    else:
        newstudy = Study(name=name, process=newproc)
    if parents:
            newstudy.add_parents(parents)
    if params:
            newstudy.add_params(params)
    newstudy.run(newstudy, local_func=process)
    newstudy.save()

# def run_study(study_id, local_func=None):
#     """Run the Study process on the inputs and return any alerts."""
#     study_obj = get_study(study_id)
#     if local_func:
#         function = local_func
#     else:
#         function = study_obj.process.function
#     parent_data = {name: study.data for name, study in study_obj.parents.items()}
#     study_obj.data, newalerts = function(**parent_data, **study_obj.params)
#     study_obj.timestamp = datetime.datetime.now(tzlocal())
#     study_obj.alertslog.record_alerts(newalerts, study_obj.timestamp)
#     study_obj.save()
#     study_obj.fire_alerts()

# def run_if(self, dummy=None):
#     """Run the Study if it's no longer valid."""
#     if not self.valid:
#         self.run()

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

