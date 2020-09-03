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
    BooleanField,
)
from mongoengine import signals
from bson.dbref import DBRef

from fintrist import util, Config
from fintrist.notify import Notification

__all__ = ('BaseStudy', 'Study', 'Backtest', 'Process', 'Trigger', 'Recipe',
    'Stream', 'Strategy')

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
    file = FileField()
    newfile = FileField()
    _timestamp = DateTimeField(default=arrow.now(Config.TZ).datetime)
    valid_age = IntField(default=0)  # Zero means always valid
    valid_type = StringField(max_length=120, default='market')

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
        return arrow.get(self._timestamp).to(Config.TZ)

    @timestamp.setter
    def timestamp(self, newdt):
        """Process the timestamp for entry."""
        if isinstance(newdt, arrow.Arrow):
            newdt = newdt.datetime
        self._timestamp = newdt

    @property
    def valid(self):
        """Check if the Study data is still valid."""
        # Check the age of the data
        if self.valid_type == 'market':
            current = self.market_valid(self.timestamp)
        elif self.valid_age == 0:
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
        schedule, nyse = util.market_schedule(timestamp, now)
        is_open = nyse.open_at_time(schedule, now.datetime)  # Market currently open
        open_close_dt = pd.DataFrame([], index=schedule.values.flatten())  ## Market day boundaries
        if is_open:  ## If the market is open, the data should be refreshed
            return False
        elif len(open_close_dt[timestamp.datetime:now.datetime]) > 0: ## A market day boundary has passed
            return False
        else:  ## The market hasn't changed. The data is still valid.
            return True

    @staticmethod
    def alert_overwrite(timestamp):
        """Check if the previous alert should be overwritten."""
        now = arrow.now(Config.TZ)
        schedule, _ = util.market_schedule(timestamp, now)
        ## New alert only if a new market day has begun. Otherwise, overwrite.
        open_dt = pd.DataFrame([], index=schedule['market_open'])
        logger.debug(open_dt)
        logger.debug(timestamp)
        logger.debug(now)
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
        parent_objects = {key: BaseStudy.objects(name=val).get() for key, val in newparents.items()}
        self.parents.update(parent_objects)
        self.save()

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
    def data(self):
        """Preprocess the data field to return the data in a usable format."""
        self.transfer_file()
        file_obj = self.file.get()
        try:
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
            if not self.file:
                self.write_to(self.file, newdata)
            else:
                self.write_to(self.newfile, newdata)
                self.transfer_file()

    def write_to(self, field, newdata):
        """Write data to a FileField."""
        field.new_file()
        field.write(pickle.dumps(newdata))
        field.close()
        self.save()

    def transfer_file(self):
        """Transfer the data from newfile to file."""
        if self.newfile:
            newfile = self.newfile.read()
            self.file.replace(newfile)
            self.save()
            self.newfile.delete()
            self.save()

    def remove_files(self):
        """Remove the data."""
        self.file.delete()
        self.newfile.delete()
        self.save(validate=False)

@clean_files.apply
class Study(BaseStudy):
    """Contains data process results."""
    # Defining the analysis that generated the data
    process = ReferenceField('Process', required=True)

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

    def set_process(self, name):
        """Set the Study's Process based on a name."""
        self.process = Process.objects(name=name).get()
        self.save()

    def update_valid_age(self, new_age):
        """Update the valid age for the data."""
        self.valid_age = new_age
        self.save()

    ## Methods related to scheduling runs ##

    def run(self, function=None, force=False):
        """Run the Study process on the inputs and return any alerts."""
        self.data, newalerts = function(**self.parents, **self.params)
        if self.alert_overwrite(self.timestamp):
            self.alertslog.remove_alert()
        self.timestamp = arrow.now(Config.TZ)
        self.alertslog.record_alerts(newalerts, self.timestamp)
        self.save()

    ## Methods for handling inputs ##

    @property
    def all_parents(self):
        """Full dict of parent kwargs, even if not set yet."""
        return {key: self.parents.get(key) for key in self.process.parents}

    @property
    def all_params(self):
        """Full dict of param kwargs, even if not set yet."""
        return {key: self.params.get(key) for key in self.process.params}

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
        return actions

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

@clean_files.apply
class Backtest(BaseStudy):
    """Contains backtesting results.

    days: Length of the backtesting interval.
    end: Last date of the backtesting interval.

    super.parents (dict):
        model: Study containing the analysis to backtest
        prices: Study containing historical price data

    run: Run the analysis on each day of the interval, recording the action
        signals that are triggered.

    trade: Simulates a trading portfolio based on the action signals stored
        after Backtest.run.
    """
    days = IntField(default=365)
    end = DateTimeField(default=arrow.now(Config.TZ).datetime)

    # pylint: disable=unsubscriptable-object
    def __repr__(self):
        return f"Backtest: {self.name}"

    @property
    def start(self):
        """The first date of the interval"""
        return self.end - dt.timedelta(days=self.days)

    def run(self, strategy, function=None):
        """Backtest the model Study on the interval and record actions."""
        model = self.parents['model']
        prices = self.parents['price']
        parent_data = {name: study.data for name, study in model.parents.items()}

        # Run on each day in the interval
        simulated = []
        tempstudy = Study()
        for view_date in model.data[self.start:self.end].index:
            trunc_data = {name: data[:view_date] for name, data in parent_data.items()}
            _, newalerts = function(**trunc_data, **model.params)
            tempstudy.alertslog.record_alerts(newalerts, view_date)
            actions = strategy.check_actions(tempstudy)
            simulated.append((view_date, actions))

        # Save the data
        simdata = pd.DataFrame(simulated, columns=['date', 'signals']).set_index('date')
        pricedata = prices.data
        simdata['price'] = (pricedata['high'] + pricedata['low'])/2
        self.data = simdata
        self.timestamp = arrow.now(Config.TZ)
        self.save()
        Notification(['printhead'], study=self, alerts=[])

class Process(Document):
    """Handles for choosing the appropriate data-processing functions.
    Parent arguments and parameters are parsed from the function docstring.
    """
    # Identity
    name = StringField(max_length=120, required=True, primary_key=True)
    local = BooleanField(default=False)

    # Args
    parents = ListField(StringField())
    params = ListField(StringField())
    alerts = ListField(StringField())

    # Meta
    schema_version = IntField(default=1)
    meta = {
        'strict': False,
        }

    def __repr__(self):
        return f"Process: {self.name}"

    def get_params(self, func):
        """Store the names for the parent data and parameter arguments."""
        parents = []
        params = []
        alerts = []
        docstr = inspect.getdoc(func)
        if docstr is None:
            return
        for line in docstr.splitlines():
            words = line.split(":: ")[-1].split(', ')
            if line.startswith('::parents::'):
                parents.extend(words)
            elif line.startswith('::params::'):
                params.extend(words)
            elif line.startswith('::alerts::'):
                alerts.extend(words)
        self.parents = parents
        self.params = params
        self.alerts = alerts

class Stream(Document):
    """A recipe for a series of sequential Study objects.
    """
    # Identity
    name = StringField(max_length=120, required=True, unique=True)

    # Args
    recipes = ListField(ReferenceField('Recipe'))
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

class Recipe(Document):
    """Recipe for a Study

    """
    # ID
    name = StringField(max_length=120, required=True, unique=True)
    studyname = StringField(max_length=120, required=True)

    # Data Inputs
    parents = MapField(StringField())  # Names of precursor data used by the Analysis
    params = DictField()  # Processing parameters.
    metaparams = ListField(StringField())  # Modifiable portions of recipe

    # Running parameters
    process = ReferenceField('Process', required=True)
    valid_age = IntField(default=0)
    valid_type = StringField(max_length=120, default='market')

    # Meta
    schema_version = IntField(default=1)
    meta = {
        'strict': False,
        }

    def __repr__(self):
        return f"Recipe: {self.name}"

    def get_metaparams(self):
        """Find all curly-bracked variables and store them."""
        searchables = [
            self.studyname,
            *self.parents.values(),
            *self.params.values(),
        ]
        new_metaparams = set()
        for case in searchables:
            new_metaparams.update(util.get_variables(case))
        self.metaparams = list(new_metaparams)
