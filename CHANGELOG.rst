#########
CHANGELOG
#########

==========
Unreleased
==========

Added
-----
* ETL for stock pricing features.
* Alpaca API.
* `market_open`

Changed
-------
* Reorganized processes into `fintrist_lib`.
* Improved .gitignore.
* Removed legacy AlphaVantage functions.

==================
0.4.1 - 2020-09-13
==================

Added
-----
* `create_sim`
* `compare_sims`
* `backtest_and_sim`
* `plot_benchmark`

Fixed
-----
* Fixed sma migration.
* Fixed create_study process input.
* Study.timestamp comes from data file metadata.
* Update `simulate` for new backtest structure.

Changed
-------
* Allow any number of years or days to be specified for backtest.
* Switch SMA analysis to use adjusted prices.
* Data from scrapers will be mocked when backtesting.

==================
0.4.0 - 2020-09-06
==================

Added
-----
* `Strategy` object, `create_strategy`, `get_strategy`.
* `fintrist.create_backtest`
* `backtest`, in fintrist_ds CATALOG.
* Migrations suite.
* Dask `close_client`.

Fixed
-----
* Timestamps weren't updating on runs.
* `create_study` now accepts strings as process names.
* `market_schedule` now tolerates empty schedules.

Changed
-------
* Transfer Triggers into new Strategy object.
* Can pass parent Study to analysis functions, instead of just data.
* Make Backtest an ordinary Study, instead of its own object.
* Triggered actions return as tuple.

==================
0.3.1 - 2020-08-25
==================

Added
-----
* Visualization library, including plot_sma.
* TO DO list.
* Tiingo stock scraper.
* Validity check against market day.

Fixed
-----
* Metaparams on Stream failed to update due to mongoengine bug.
* Objects need to be reloaded after Document.update().

Changed
-------
* Study.alerts now shows newactive and newinactive as well.
* Alerts are now overwritten if a new market day has not started.

==================
0.3.0 - 2020-08-09
==================

Added
-----
* Services in fintrist for creating and manipulating database objects.
* Services in fintrist_ds for running and scheduling studies.
* Recipe and Stream objects for templating Studies.

Changed
-------
* Split app up into microservices architecture.
* Scheduler is now a separate package, crontris.
* fintrist_ds now handles dask and all data processing.

==================
0.2.0 - 2019-10-19
==================

Added
-----
* moving_avg
* Backtest
* simulate
* multisim

Changed
-------
* Split processes off to fintrist_ds subpackage.

==================
0.1.1 - 2019-06-23
==================

Added
-----
* Dask processing
* Dash app (fintrist_app v2)

Changed
-------
* Removed Stream model.
* Implemented dependency resolution at the Study level.

==================
0.1.0 - 2019-06-06
==================

Added
-----
* MongoDB backend for data storage.
* fintrist_app
* APScheduler

==================
0.0.1 - 2018-03-23
==================

Added
-----
* Stock indicators

Changed
-------
* Switched to Alpha Vantage stock data.

==================
0.0.0 - 2016-12-12
==================

Added
-----
* fintrist origin
