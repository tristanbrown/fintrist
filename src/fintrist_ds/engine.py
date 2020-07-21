"""
The engine that applies analyses to studies.
"""
from fintrist import get_study, get_process, Process, Study

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
    newstudy.run(local_func=process)
    newstudy.save()
