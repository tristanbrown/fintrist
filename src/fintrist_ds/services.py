"""Helpful summary functions"""
import pandas as pd

from fintrist import get_study

def compare_sims(sim1, sim2=None):
    sim1 = get_study(sim1)
    if sim2 is None:
        symbol = sim1.name.split(' ')[0]
        sim2 = f"{symbol} daily, buy and hold Sim"
    sim2 = get_study(sim2)
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
