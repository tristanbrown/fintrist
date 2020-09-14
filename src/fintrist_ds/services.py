"""Helpful summary functions"""
import pandas as pd

from fintrist import get_study, create_backtest, create_sim
from .engine import schedule_study

def compare_sims(sim1, sim2=None):
    sim1 = get_study(sim1)
    if sim2 is None:
        symbol = sim1.name.split(' ')[0]
        sim2 = f"{symbol} daily, buy and hold Sim"
    sim2 = get_study(sim2)
    if sim2 is None:
        sim2 = backtest_and_sim(symbol, f"{symbol} daily", "buy and hold")
    comparison = pd.merge(
        how='left',
        left=sim1.data,
        right=sim2.data,
        left_index=True,
        right_index=True
    )
    comparison['rel profit'] = comparison['value_x'] - comparison['value_y']
    comparison['rel % return'] = comparison['% return_x'] - comparison['% return_y']
    return comparison

def backtest_and_sim(symbol, model, strategy, period='5y'):
    backtest = create_backtest(model, strategy, period)
    sim = create_sim(symbol, backtest.name)
    schedule_study(sim, force=True)
    sim.reload()
    return sim
