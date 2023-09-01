from circuitree.parallel import ParameterTable, TranspositionTable
from dask.distributed import Client, LocalCluster
from gillespie import PARAM_NAMES
from oscillation_parallel_dask import OscillationTreeDask
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    n_workers = 50

    print("Loading parameter table")
    param_sets_csv = Path(
        "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000_5tf.csv"
    ).expanduser()
    param_df = pd.read_csv(param_sets_csv, index_col="sample_num")
    param_df.index.name = "visit"

    param_table = ParameterTable.from_dataframe(
        param_df,
        seed_col="visit",
        init_cols=["A_0", "B_0", "C_0", "D_0", "E_0"],
        param_cols=PARAM_NAMES,
    )

    save_dir = Path(
        "~/git/circuitree-paper/data/oscillation/mcts/230831_dask"
    ).expanduser()
    save_dir.mkdir(exist_ok=True)

    print("Making Dask Client")
    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster, direct_to_workers=True)
    print(client)

    print("Making CircuiTree object")
    tree = OscillationTreeDask(
        root="ABCDE::",
        components=["A", "B", "C", "D", "E"],
        interactions=["activates", "inhibits"],
        dt=20.0,  # seconds
        nt=2000,
        # batch_size=50,
        batch_size=n_workers,
        parameter_table=param_table,
        save_dir=save_dir,
        cluster=cluster,
        save_ttable_every=10,
    )
    print("Starting MCTS")
    tree.search_mcts(1_000_000)

    ...
