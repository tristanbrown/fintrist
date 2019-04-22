"""
Database Management
"""
import sys
import inspect
import hashlib

from mongoengine.errors import NotUniqueError
from fintrist import processes, util
from fintrist.models import Process

def register():
    """Register all of the processes in the database."""
    new_procs = processes.ALL
    for name in new_procs:
        # Check against most recent entry with matching name
        recent = get_newest_proc(name)
        new_fingprnt = encode_func(new_procs[name])
        if recent and new_fingprnt == recent.fingerprint:
            print("'{}' skipped: Already registered.".format(name))
            continue
        elif recent:
            new_version = recent.version + 1
        else:
            new_version = 1
        # Save the new Process
        parents, params = get_proc_params(name)
        Process(
            name=name,
            fingerprint=new_fingprnt,
            version=new_version,
            parents=parents,
            params=params,
            ).save()
        print("Inserted '{}'.".format(name))

def clear():
    """Delete all processes in the database."""
    Process.drop_collection()
    print("Cleared the processes database.")

def get_newest_proc(name):
    """Return the most recent version of a process."""
    return Process.objects(name=name).order_by("-version").limit(-1).first()

def get_proc_params(name):
    """Return the names for the parent data and parameter arguments."""
    source = processes.ALL[name]
    docstr = inspect.getdoc(source)
    # TODO: get the parameters from the docstring
    parents = ['parent1', 'parent2']
    params = ['param1', 'param2']
    return parents, params

def encode_func(func):
    """Encode a function as a hash."""
    return hashlib.sha1(str.encode(inspect.getsource(func))).hexdigest()


@util.not_implemented
def test():
    print("Test function")

def main():
    """Run the function specified by the command line argument."""
    arg = sys.argv[1]
    globals()[arg]()

if __name__ == "__main__":
    main()
