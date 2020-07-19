"""Helper functions."""
import logging
from mongoengine.errors import SaveConditionError, DoesNotExist
from .models import Study, BaseStudy

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

def get_study(name):
    """Get a certain Study name or BaseStudy by name."""
    try:
        return BaseStudy.objects(name=name).get()
    except DoesNotExist:
        logger.debug(f"Study '{name}' does not exist.")

def get_data(name):
    """Get the data by a certain Study name or BaseStudy name."""
    obj = get_study(name)
    if obj:
        return obj.data
