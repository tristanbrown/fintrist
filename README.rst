========
Fintrist
========
A data analysis engine focused on time-dependent data. 

Install
=======

To install:

- Install Docker Desktop.
- Pull the microservice images.

Setup
=====

To set up the microservices:

- Run ``docker-compose up -d``
- This will start the docker network with containers for rabbitmq, mongodb,
  monbodb-backup, and crontris.
- Use ``docker ps -a`` to check that all of the containers started correctly.

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
