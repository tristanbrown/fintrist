========
Fintrist
========
A data analysis engine focused on time-dependent data. 

Install
=======

Coming soon

Studies
=======

Coming soon

Backtesting
===========

To backtest a model:

- Create a new Backtest object with two parent Studies:
  ``model`` and ``prices``. 
- Use ``Backtest.run`` to generate the action signals.
- Create a new Study with the ``'simulate'`` process,
  with the Backtest as a parent. 
- Run the simulation Study to generate data representing a portfolio.
- Instead, use the ``'multisim'`` process to attempt the simulation across
  multiple time intervals, tabulating the returns. 
