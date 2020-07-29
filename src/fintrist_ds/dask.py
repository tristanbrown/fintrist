from dask.distributed import Client, LocalCluster

try:
    ## Connect to existing cluster
    client = Client('127.0.0.1:8786')
except OSError:
    ## Create new cluster
    cluster = LocalCluster(ip="127.0.0.1", scheduler_port=8786, processes=True)
    client = Client(cluster)

## - Also make sure study.run functions will serialize
