"""Helper functions."""
import logging
from mongoengine.errors import SaveConditionError, DoesNotExist
from .models import Study, BaseStudy, Process

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

def get_study(study_id):
    """Get a certain Study name or BaseStudy by name."""
    if isinstance(study_id, Study):
        return study_id
    elif isinstance(study_id, str):
        try:
            return BaseStudy.objects(name=study_id).get()
        except DoesNotExist:
            logger.debug(f"Study '{study_id}' does not exist.")

def get_data(name):
    """Get the data by a certain Study name or BaseStudy name."""
    obj = get_study(name)
    if obj:
        return obj.data

def get_process(name):
    """Get a certain Study name or BaseStudy by name."""
    try:
        return Process.objects(name=name).get()
    except DoesNotExist:
        logger.debug(f"Process '{name}' does not exist.")
