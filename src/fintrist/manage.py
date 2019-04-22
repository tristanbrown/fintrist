"""
Database Management
"""
import sys

from mongoengine.errors import NotUniqueError
from fintrist import processes, util
from fintrist.models import Process

def register():
    """Register all of the processes in the database."""
    new_procs = processes.ALL
    for name in new_procs:
        new_function = new_procs[name]
        try:
            Process(
                name=name,
                function=new_function,
                ).save(force_insert=True)
            print("Inserted '{}'.".format(name))
        except NotUniqueError:
            print("'{}' skipped: Already registered.".format(name))

def clear():
    """Delete all processes in the database."""
    Process.drop_collection()
    print("Cleared the processes database.")

@util.not_implemented
def test():
    print("Test function")

def main():
    """Run the function specified by the command line argument."""
    arg = sys.argv[1]
    globals()[arg]()

if __name__ == "__main__":
    main()
