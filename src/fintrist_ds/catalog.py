import sys
import inspect as ins
import logging

import arrow
import pandas as pd

from fintrist import Study
from .settings import Config
from .scrapers import base
from . import analysis

logger = logging.getLogger(__name__)

## The process registry ##

# CATALOG = dict(ins.getmembers(sys.modules[__name__], ins.isfunction))
SCRAPERS_CATALOG = dict(ins.getmembers(base, ins.isfunction))
ANALYSIS_CATALOG = dict(ins.getmembers(analysis, ins.isfunction))
CATALOG = {**SCRAPERS_CATALOG, **ANALYSIS_CATALOG}

def backtest(model, strategy, period='1y', end=None):
    """Run the model Study on previous dates over the period,
    collecting the alerts.
    ::parents:: model
    ::params:: strategy, period, end
    ::alerts:: complete
    """
    # Define the time period
    if not end:
        end = arrow.now(Config.TZ)
    if period == '1y':
        start = end.shift(years=-1)
    elif period == '10d':
        start = end.shift(days=-10)
    else:
        start = end.shift(years=-100)

    # Set up the fake study to run
    simulated = []
    tempstudy = Study()
    parent_data = {name: study.data for name, study in model.parents.items()}
    function = CATALOG[model.process.name]

    # At each date, run the model's function on the previous data
    for view_date in model.data[start.date():end.date()].index:
        print(f"Backtesting at {view_date}")
        logger.info(f"Log: Backtesting at {view_date}")
        prev_date = arrow.get(view_date).to(Config.TZ).shift(days=-1)
        trunc_data = {name: data[:prev_date.date()] for name, data in parent_data.items()}
        _, newalerts = function(**trunc_data, **model.params)
        tempstudy.alertslog.record_alerts(newalerts, view_date)
        actions = strategy.check_actions(tempstudy)
        simulated.append((view_date, actions))

    # Save the data
    simdata = pd.DataFrame(simulated, columns=['date', 'signals']).set_index('date')
    return simdata, ['complete']

CATALOG['backtest'] = backtest
