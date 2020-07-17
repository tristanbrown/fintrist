"""Helper functions."""

from .models import Study, BaseStudy

def store_data(data, name, overwrite=False):
    """Create a new BaseStudy with the given data."""
    try:
        archive = BaseStudy(name=name, data=data)
        archive.save()
    except:
        if overwrite:
            print("Overwrite not implemented")
        else:
            raise

def get_data(name):
    """Get the data by a certain Study name or BaseStudy name."""
    obj = Study.objects(name=name).get()
    return obj.data
