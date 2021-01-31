#########
CHANGELOG
#########

==========
Unreleased
==========

Added
-----
* In learn.py, DFData does train/test split.
* NN Trainer does validation on holdout test data set.
* Can now transfer Study data to an archive dict.
* Add method for deleting orphaned GridFS file data.
* Add NN prediction function.

Fixed
-----
* Fix populating NN performance df.
* Update Pytorch dependency to >=1.6 to fix random_split seeding.
* Preserve null values in TrendLengthData.

Changed
-------
* Improve NN metrics printing.
* Feature reduction in TrendLengthData.
* Generalize Study file transfer method.
* Expose db pointer in fintrist.
* Clarify dataset variable names.

==================
1.1.0 - 2021-01-21
==================

Added
-----
* Can toggle into a Test database.
* Add NNModel study type.
* Add create_nn for creating NN Study objects.
* Add learn.py to fintrist_lib, with NN creation and manipulation functions.

Changed
-------
* Expose fintrist.mongoclient.
* Enable get_study to find studies by recipe/param combos.

==================
1.0.0 - 2020-12-05
==================

Added
-----
* Recipes can spawn parent studies.
* Study objects can find their own Recipes.
* `generate` function to spawn studies and immediately schedule them.
* `generate_all` function to generate studies on multiple symbols in parallel.

Fixed
-----
* Allow `create_study` to overwrite previous process, parents, and params.
* Eliminate all `Document.reload` to avoid bug that breaks FileField.
* Set dask scheduler to treat processes as not "pure". 

Changed
-------
* Merge `Process` and `Recipe` in fintrist_lib.
* Rename `stock` process to `stock_daily`.
* Removed manage.register.
* Generalized manage.clear.
* Move `get_recipe` to fintrist_lib.
* Allow `get_recipe` to deliver Recipe objects from Recipes or functions.

==================
0.5.0 - 2020-10-17
==================

Added
-----
* ETL for stock pricing features.
* Alpaca API.
* `market_open`
* `stock_intraday` to get stock history at 1min intervals.

Fixed
-----
* Limit CATALOG to functions specified in `__all__`.

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
