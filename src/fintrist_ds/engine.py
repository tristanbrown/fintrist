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

__all__ = ['test_func', 'build_dag', 'schedule_study', 'store_result',
    'get_function', 'dsk']

def test_func(*args):
    print(f"Received {args}")

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
    a_study = get_study(a_study)
    dag = build_dag(a_study, force)
    print(dag)
    client.get(dag, str(a_study.id), num_workers=Config.NUM_WORKERS)
    # from time import time
    # start = time()
    # client.get(dsk, 'f5')
    # duration = time() - start
    # print(duration)
    # return duration

def store_result(name, process, parents=None, params=None, **kwargs):
    """Use a local or library function to create and run a new Study."""
    if isinstance(process, str):
        process = get_function(process)
    newstudy = create_study(name, process, parents, params, **kwargs)
    newstudy.run(function=process)
    newstudy.save()
    return newstudy

# Testing below

class Obj():
    def __init__(self, name, delay, inputs=None):
        self.name = name
        self.delay = delay
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
    
    def wait(self, dummy):
        from time import sleep
        print(f"{self.name}: Start waiting for {self.delay} sec")
        sleep(self.delay)
        print(f"{self.name}: Done waiting for {self.delay} sec")
    
db = {
    'f1': Obj('f1', 10),
    'f2': Obj('f2', 20),
    'f3': Obj('f3', 5, ['f1', 'f2']),
    'f4': Obj('f4', 30),
    'f5': Obj('f5', 1, ['f3', 'f4'])
}
dsk = {key: (val.wait, val.inputs) for key, val in db.items()}
