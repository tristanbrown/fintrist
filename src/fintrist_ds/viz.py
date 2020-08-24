"""Visualizations"""
import datetime

from fintrist import get_data, get_study, spawn_stream
from .engine import schedule_study

def plot_stream(str_name, window='1y', **kwargs):
    study = spawn_stream(str_name, **kwargs)[-1]
    schedule_study(study.name, force=False)  # NOTE: check if this is finished?
    study = get_study(study.name)  # Refresh the study with new data.
    print(f"Alerts: {study.alerts}")
    data = study.data
    if window == '1y':
        today = datetime.datetime.now().date()
        last_year = today - datetime.timedelta(weeks=52)
        data = data[last_year:today]
    data.plot(title=study.name)

def plot_sma(symbol, window='1y'):
    return plot_stream('Stock SMA', window, symbol=symbol)
