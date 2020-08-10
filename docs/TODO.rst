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
    within the `valid_age`.
* Make sure Backtest still works.

Change
------
* Make `simulate` work with the new Strategy object system.
