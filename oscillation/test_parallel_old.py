from dask.distributed import Client, LocalCluster
import numpy as np
from pathlib import Path
import search_parallel as pl

param_names = [
    "k_on",
    "k_off_1",
    "k_off_2",
    "km_unbound",
    "km_act",
    "km_rep",
    "km_act_rep",
    "kp",
    "gamma_m",
    "gamma_p",
]
init_names = ["A_0", "B_0", "C_0"]
nt = 2000
dt = 20.0
time_points = np.linspace(0, nt * dt, nt, endpoint=False)

# save_dir = Path("../../data/oscillation/tmp").resolve().absolute()
save_dir = Path("/tmp/data/circuitree/oscillation")

components = ["A", "B", "C"]
interactions = ["activates", "inhibits"]
root = "ABC::"


if __name__ == "__main__":
    # cluster = LocalCluster(n_workers=12, threads_per_worker=1)
    # client = Client(cluster)
    client = ...
    # print(client.dashboard_link)

    master_tree = pl.MasterTree(
        components=components,
        interactions=interactions,
        root=root,
        time_points=time_points,
        save_dir=save_dir,
        client=client,
        init_condition_names=init_names,
        param_names=param_names,
        parallel_level=1,
        success_threshold=0.005,
        autocorr_threshold=0.5,
        init_mean=10.0,
        extension="parquet",
    )
    master_tree.spawn_parallel_trees(depth=1)
    master_tree.search_parallel(200)

    ...
