import sys
import logging

import arrow
import pandas as pd

from fintrist import Study
from fintrist_lib import ANALYSIS_CATALOG, SCRAPERS_CATALOG
from .settings import Config
from . import util

logger = logging.getLogger(__name__)

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
    quant, unit = util.split_alphanum(period)
    if unit == 'y':
        start = end.shift(years=-quant)
    elif unit == 'd':
        start = end.shift(days=-quant)
    else:
        start = end.shift(years=-100)

    # Set up the fake study to run
    simulated = []
    tempstudy = Study()
    parent_data = {name: study.data for name, study in model.parents.items()}
    try:
        recipe = ANALYSIS_CATALOG[model.recipe]
    except KeyError:
        recipe = SCRAPERS_CATALOG[model.recipe]
        parent_data['mock'] = model.data

    # At each date, run the model's process on the previous data
    # TODO: Date range should be based on the model's parents, not the model.
    full_range = model.data.index
    for view_date in model.data[start.date():end.date()].index:
        logger.debug(f"Log: Backtesting at {view_date}")
        curr_idx = full_range.get_loc(view_date)
        try:
            prev_date = full_range[curr_idx - 1]
        except IndexError:
            continue
        trunc_data = {name: data[:prev_date] for name, data in parent_data.items()}
        _, newalerts = recipe.process(**trunc_data, **model.params)
        tempstudy.alertslog.record_alerts(newalerts, view_date)
        actions = strategy.check_actions(tempstudy)
        simulated.append((view_date, actions))

    # Save the data
    simdata = pd.DataFrame(simulated, columns=['date', 'signals']).set_index('date')
    return simdata, ['complete']
