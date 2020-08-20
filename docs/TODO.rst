#####
TO DO
#####

Add
---
* Create a Strategy object to host Triggers separate from the Studies.
* Implement a neural network-training Study. 

Fix
---
* Alerts should overwrite the latest one if the Study is rerun
    within the `valid_age`. (test this)
* Make sure Backtest still works.

Change
------
* Make `simulate` work with the new Strategy object system.
* Set `valid` property based on the trading day, not just a time period.
* Consider incorporating current market price in 'stock' data, not just `close`.
