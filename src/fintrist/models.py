"""
The engine that applies analyses to data and generates alerts.
"""
import logging
import pickle
import inspect
import datetime as dt
import arrow

import numpy as np
import pandas as pd
from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (
    DateTimeField, DictField, EmbeddedDocumentField,
    EmbeddedDocumentListField, IntField, FileField,
    ListField, MapField, ReferenceField, StringField,
    BooleanField, BinaryField, GridFSProxy,
)
from pymongo.errors import InvalidDocument
from mongoengine import signals
from bson.dbref import DBRef

from fintrist import util, Config
from fintrist.notify import Notification
from fintrist_lib import get_recipe, learn
from fintrist_lib.scrapers.stockmarket import market_schedule

__all__ = ('BaseStudy', 'Study', 'Trigger', 'Stream', 'Strategy')

logger = logging.getLogger(__name__)

class Alerts(EmbeddedDocument):
    """Alerts generated by a Study."""
    _timestamp = DateTimeField(default=arrow.now(Config.TZ).datetime)
    active = ListField(StringField(max_length=120))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    @property
    def timestamp(self):
        """Preprocess the timestamp to ensure consistency."""
        return arrow.get(self._timestamp).to(Config.TZ)

    @timestamp.setter
    def timestamp(self, newdt):
        """Process the timestamp for entry."""
        if isinstance(newdt, arrow.Arrow):
            newdt = newdt.datetime
        self._timestamp = newdt

class AlertsLog(EmbeddedDocument):
    """Log of Alerts."""
    log = EmbeddedDocumentListField('Alerts')
    count = IntField(default=100)

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    def trim(self):
        """Trim the log to the specified size."""
        if len(self.log) > self.count:
            self.log = self.log[:self.count]

    def record_alerts(self, alerts, timestamp):
        """Take in a new list of alerts.

        alerts (list): [str]
        """
        newalerts = Alerts(active=alerts)
        newalerts.timestamp = timestamp
        self.log = [newalerts] + self.log
    
    def remove_alert(self, idx=0):
        """Delete the alert at the index. First by default."""
        if self.log:
            self.log.pop(idx)

    def get_alerts(self, idx):
        """Get the alerts at the given lookback index."""
        try:
            return set(self.log[idx].active)
        except IndexError:
            return set()

    @property
    def newest(self):
        """Most recent alerts."""
        return self.get_alerts(0)

    @property
    def newactive(self):
        """Newly active alerts."""
        return self.get_alerts(0) - self.get_alerts(1)

    @property
    def newinactive(self):
        """Newly inactive alerts."""
        return self.get_alerts(1) - self.get_alerts(0)

    def clear(self):
        """Delete all alerts."""
        self.log = []

class Trigger(EmbeddedDocument):
    """A rule determining how an action is triggered."""
    alert_types = ('all', 'active', 'inactive')
    action_choices = ('buy', 'sell', 'positive', 'negative')
    on = StringField(default='active', choices=alert_types)
    # TODO: Change matchtext to populate choices based on the analysis
    matchtext = StringField(max_length=120)
    actions = ListField(StringField(choices=action_choices))

    # Meta
    schema_version = IntField(default=1)
    meta = {'strict': False}

    def __repr__(self):
        return f"Trigger: {str(self)}"

    def __str__(self):
        return f"{self.matchtext} {self.on}"

    def check_fire(self, study):
        """Check if the trigger should be fired, and then run actions."""
        triggered = self.check_conds(study.alertslog)
        if triggered:
            self.fire(study=study, alerts=triggered)

    def get_actions(self, alertslog):
        """Return any actions that should be triggered."""
        triggered = self.check_conds(alertslog)
        if triggered:
            return self.actions
        else:
            return []

    def check_conds(self, alertslog):
        """Check if the conditions for triggering have been met by the alert."""
        if self.on == 'active':
            alerts = alertslog.newactive
        elif self.on == 'inactive':
            alerts = alertslog.newinactive
        elif self.on == 'all':
            alerts = alertslog.newest
        triggered = [alert for alert in alerts if self.matchtext == alert]
        return triggered

    def fire(self, **kwargs):
        """Trigger the specified actions."""
        Notification(self.actions, **kwargs)

@util.handler(signals.pre_delete)
def clean_files(sender, document):  #pylint: disable=unused-argument
    """Signal deleted Studies to remove data files."""
    document.remove_files()

@clean_files.apply
class BaseStudy(Document):
    """Contains data process results.

    Can act as a generic data archive.
    """
    # ID
    name = StringField(max_length=120, required=True, unique=True)

    # Data Inputs
    parents = MapField(ReferenceField('BaseStudy'))  # Precursor data used by the Analysis
    params = DictField()  # Processing parameters.

    # Data Outputs
    newfile = MapField(FileField())
    fileversions = MapField(FileField())
    _timestamp = StringField()
    valid_age = IntField(default=0)  # Zero means always valid
    valid_type = StringField(choices=['market', 'always'], default='market')

    # Run Status
    status = StringField(choices=['Idle', 'Running'], default='Idle')

    # Meta
    schema_version = IntField(default=1)
    meta = {
        'strict': False,
        'collection': 'study',
        'allow_inheritance': True,
        }

    def __repr__(self):
        return f"BaseStudy: {self.name}"

    # pylint: disable=no-member
    def clean(self):
        """Clean attributes."""
        # Parents
        for key, parent in self.parents.items():
            if isinstance(parent, DBRef):
                del self.parents[key]
        # Timestamp display
        try:
            self._timestamp = self.timestamp.format()
        except AttributeError:
            self._timestamp = None
        # Subclassed cleaning
        self.subclean()

    def subclean(self):
        """Cleaning operations for subclasses."""
        pass

    ## Methods defining the Study ##

    def rename(self, newname):
        """Rename the Study."""
        self.name = newname
        self.save()

    ## Methods related to scheduling runs ##

    @property
    def timestamp(self):
        """Preprocess the timestamp to ensure consistency."""
        return self.get_timestamp('default')

    def get_timestamp(self, version):
        try:
            recent_file = self.fileversions.get(version)
            return arrow.get(recent_file.uploadDate).to(Config.TZ)
        except:
            return

    @property
    def valid(self):
        """Check if the Study data is still valid."""
        # Check the age of the data
        if not self.timestamp:
            current = False
        elif self.valid_type == 'market':
            current = self.market_valid(self.timestamp)
        elif self.valid_type == 'always' or self.valid_age == 0:
            current = True
        else:
            current = arrow.now(Config.TZ) - self.timestamp < dt.timedelta(days=self.valid_age)
        # Check if the parents are valid too
        if current:
            for parent in self.parents.values():
                if not parent.valid:
                    current = False
                    break
        return current

    @staticmethod
    def market_valid(timestamp):
        """Check if the market has or hasn't progressed since the last timestamp."""
        now = arrow.now(Config.TZ)
        schedule, nyse = market_schedule(timestamp, now)
        is_open = nyse.open_at_time(schedule, now.datetime)  # Market currently open
        open_close_dt = pd.DataFrame([], index=schedule.values.flatten())  ## Market day boundaries
        if is_open:  ## If the market is open, the data should be refreshed
            return False
        elif len(open_close_dt[timestamp.datetime:now.datetime]) > 0: ## A market day boundary has passed
            return False
        else:  ## The market hasn't changed. The data is still valid.
            return True

    @staticmethod
    def alert_overwrite(timestamp, now):
        """Check if the previous alert should be overwritten."""
        if timestamp is None:
            return False
        schedule, _ = market_schedule(timestamp, now)
        ## New alert only if a new market day has begun. Otherwise, overwrite.
        open_dt = pd.DataFrame([], index=schedule['market_open'])
        return len(open_dt[timestamp.datetime:now.datetime]) == 0
    
    @property
    def dependencies(self):
        """Create a dictionary of dependencies."""
        deps = {str(self.id): [str(parent.id) for parent in self.parents.values()]}
        for parent in self.parents.values():
            deps.update(parent.dependencies)
        return deps

    def run_if(self, function=None):
        """Run the Study if it's no longer valid."""
        if not self.valid:
            self.run(function)

    def run(self, function=None, force=False):
        """Run the Study process on the inputs and return any alerts."""
        raise Exception("Cannot run from BaseStudy objects.")

    ## Methods for handling inputs ##

    def add_parents(self, newparents):
        """Add all of the parents in the given dict of ids."""
        try:
            parent_objects = {key: BaseStudy.objects(name=val).get() for key, val in newparents.items()}
        except InvalidDocument:
            parent_objects = newparents
        self.parents.update(parent_objects)
        self.save()

    def set_parents(self, newparents):
        """Overwrite the existing parents with new ones."""
        self.parents = {}
        self.add_parents(newparents)

    def add_params(self, newparams):
        """Add all of the params in the given dict."""
        self.params.update(newparams)
        self.save()

    def remove_inputs(self, inputs):
        """Remove all of the inputs in the given iterable of names."""
        for key in inputs:
            self.parents.pop(key, None)
            self.params.pop(key, None)
        self.save()

    ## Methods for handling the saved data ##

    @property
    def version(self):
        return getattr(self, '_version', 'default')

    @version.setter
    def version(self, label):
        self._version = label

    @property
    def all_versions(self):
        return list(self.fileversions.keys())

    @property
    def data(self):
        """Preprocess the data field to return the data in a usable format."""
        if self.newfile.get(self.version):
            self.transfer_file(self.newfile, self.fileversions)
        try:
            file_obj = self.fileversions.get(self.version).get()
            result = file_obj.read()
            file_obj.seek(0)
            return pickle.loads(result)
        except:
            return None

    @data.setter
    def data(self, newdata):
        """Process the data for storage."""
        if newdata is None:
            self.remove_files()
        else:
            self.write_version(self.newfile, newdata)
            self.transfer_file(self.newfile, self.fileversions)

    def write_to(self, field, newdata):
        """Write data to a FileField."""
        field.new_file()
        field.write(pickle.dumps(newdata))
        field.close()
        self.save()

    def write_version(self, field, newdata):
        """Write data into a mapped FileField."""
        fileslot = self.get_fileslot(field)
        self.write_to(fileslot, newdata)

    def get_fileslot(self, field):
        """Get an existing fileslot in a mapfield, or create it."""
        fileslot = field.get(self.version, GridFSProxy())
        field[self.version] = fileslot
        return fileslot

    def copy_file(self, filesrc, filedest):
        """Copy the data from filesrc to filedest."""
        newfile = filesrc.read()
        filedest.replace(newfile)
        self.save()

    def transfer_file(self, filesrc, filedest):
        """Transfer a file between FileFields, possibly within a MapField."""
        try:
            filesrc = filesrc.pop(self.version, None)
        except AttributeError:
            pass
        if isinstance(filedest, dict):
            filedest = self.get_fileslot(filedest)
        self.copy_file(filesrc, filedest)
        filesrc.delete()
        self.save()

    def remove_file(self, field):
        """Remove a file version from a MapField."""
        field[self.version].delete()
        del field[self.version]
        self.save()

    def remove_files(self):
        """Remove the data."""
        for field in (self.fileversions, self.newfile):
            try:
                self.remove_file(field)
            except KeyError:
                pass

@clean_files.apply
class Study(BaseStudy):
    """Contains data process results."""
    # Defining the analysis that generated the data
    recipe = StringField(max_length=120, required=True)

    # Alerts
    alertslog = EmbeddedDocumentField('AlertsLog', default=AlertsLog())

    # pylint: disable=no-member
    # pylint: disable=not-a-mapping
    # pylint: disable=unsupported-delete-operation
    # pylint: disable=unsupported-assignment-operation
    def __repr__(self):
        return f"Study: {self.name}"

    def subclean(self):
        """Clean attributes."""
        # Alertslog
        self.alertslog.trim()

    ## Methods defining the Study ##

    def get_recipe(self):
        """Get the Study's Recipe based on a name."""
        return get_recipe(self.recipe)

    def get_process(self):
        """Get the Study's process."""
        return self.get_recipe().process

    def update_valid_age(self, new_age):
        """Update the valid age for the data."""
        self.valid_age = new_age
        self.save()

    ## Methods related to scheduling runs ##

    def run(self, function=None, force=False):
        """Run the Study process on the inputs and return any alerts."""
        prev_timestamp = self.timestamp
        if function is None:
            function = self.get_process()
        self.status = 'Running'
        try:
            parent_data = {name: parent.data for name, parent in self.parents.items()}
            self.data, newalerts = function(**parent_data, **self.params)
        except AttributeError as ex:
            if "object has no attribute 'parents'" in str(ex):
                self.data, newalerts = function(**self.parents, **self.params)
            else:
                raise
        finally:
            self.status = 'Idle'
        if self.alert_overwrite(prev_timestamp, self.timestamp):
            self.alertslog.remove_alert()
        self.alertslog.record_alerts(newalerts, self.timestamp)
        self.save()

    ## Methods for handling inputs ##

    @property
    def all_parents(self):
        """Full dict of parent kwargs, even if not set yet."""
        return {key: self.parents.get(key) for key in self.recipe.parents}

    @property
    def all_params(self):
        """Full dict of param kwargs, even if not set yet."""
        return {key: self.params.get(key) for key in self.recipe.params}

    ## Methods for handling alerts ##

    @property
    def alerts(self):
        """Most recent alerts."""
        return {
            'current': self.alertslog.newest,
            'newactive': self.alertslog.newactive,
            'newinactive': self.alertslog.newinactive,
        }

    def clear_log(self):
        """Remove log entries."""
        self.alertslog.clear()
        self.save()

@clean_files.apply
class NNModel(BaseStudy):
    """Contains neural network parameters."""
    valid_type = 'always'
    target_col = StringField(max_length=120)

    def __repr__(self):
        return f"NN: {self.name}"

    @property
    def dataset(self):
        return self.parents['dataset'].data

    @property
    def trainer(self):
        return learn.Trainer(self.dataset, self.target_col, **self.data)

    def switch_net(self, depth, width, outputs, output_type):
        trainer = self.trainer
        trainer.switch_net(depth, width, outputs, output_type)
        self.data = trainer.state

    def update_state(self, stateargs):
        trainer = self.trainer
        trainer.update_state(stateargs)
        self.data = trainer.state
        return trainer

    def train(self, epochs=10, save_interval=5, restart=False, **stateargs):
        if restart:
            self.reset()
        trainer = self.update_state(stateargs)
        print(trainer.net)
        print("Batch size: ", trainer.batch_size)
        print("Max LR: ", trainer.state['scheduler']['max_lrs'][0])
        print("Gamma: ", trainer.state['scheduler']['gamma'])
        total_epochs = trainer.epoch + epochs
        while trainer.epoch < total_epochs:
            trainer.train(save_interval)
            self.data = trainer.state

    def reset(self):
        self.data = {}

    def run(self, **kwargs):
        try:
            self.status = 'Running'
            self.save()
            self.train(**kwargs)
        finally:
            self.status = 'Idle'
            self.save()

    def predict(self, count=None):
        inputs = self.dataset.drop(self.target_col, axis=1)
        if count:
            inputs = inputs.tail(count)
        return self.trainer.predict(inputs)

class Strategy(Document):
    """A set of triggers for market actions.

    A Strategy should take in a Study, examine its alerts, and determine
    whether to give a buy or sell signal.

    A good Strategy will generate profit, when heeded.
    """
    # ID
    name = StringField(max_length=120, required=True, unique=True)
    triggers = MapField(EmbeddedDocumentField('Trigger'))

    # Meta
    schema_version = IntField(default=1)
    meta = {
        'strict': False,
        }

    def __repr__(self):
        return f"Strategy: {self.name}"

    def fire_alerts(self, study):
        """Fire alert triggers based on newly active and inactive alerts."""
        for trigger in self.triggers.values():
            trigger.check_fire(study)

    def check_actions(self, study):
        """Check the triggered actions."""
        actions = set()
        logger.debug(study.alerts)
        for trigger in self.triggers.values():
            actions.update(trigger.get_actions(study.alertslog))
        return tuple(actions)

    def get_trigger(self, trig_id):
        """Return the desired trigger."""
        return self.triggers.get(trig_id)

    def add_trigger(self, matchtext, **kwargs):
        """Add a Trigger to the Study, or update a matching one."""
        new = Trigger(matchtext=matchtext, **kwargs)
        self.triggers[str(new)] = new
        self.save()

    def del_trigger(self, trig_id):
        """Delete the specified trigger."""
        try:
            del self.triggers[trig_id]
            self.save()
        except KeyError:
            print(f"Trigger '{trig_id}' not found.")

class Stream(Document):
    """A recipe for a series of sequential Study objects.
    """
    # Identity
    name = StringField(max_length=120, required=True, unique=True)

    # Args
    recipes = ListField(StringField(max_length=120))
    metaparams = DictField()  # All Recipe's metaparams. Can be None or filled in.

    # Meta
    schema_version = IntField(default=1)
    meta = {
        'strict': False,
        }

    def __repr__(self):
        return f"Stream: {self.name}"

    def get_metaparams(self):
        all_metaparams = set()
        for recipe in self.recipes:
            all_metaparams.update(recipe.metaparams)
        updated_params = {  ## Can't update one at a time (mongoengine bug)
            recipe_param: self.metaparams.get(recipe_param)
            for recipe_param in all_metaparams}
        self.metaparams = updated_params

    def fill_metaparam(self, key, value):
        self.metaparams[key] = value
