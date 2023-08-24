from circuitree.parallel import ParameterTable
from oscillation_parallel_celery import OscillationTreeCelery
import pandas as pd
from pathlib import Path

param_sets_csv = Path(
    "/home/pbhamidi/git/circuitree-paper/data/oscillation/param_sets_queue_10000.csv"
)
param_df = pd.read_csv(param_sets_csv, index_col="sample_num")
param_df.index.name = "visit"

param_table = ParameterTable.from_dataframe(
    param_df,
    seed_col="visit",
    init_cols=["A_0", "B_0", "C_0"],
    param_cols=[
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
    ],
)

save_dir = Path(
    "/home/pbhamidi/git/circuitree-paper/data/oscillation/mcts/230823_celery"
)

tree = OscillationTreeCelery(
    root="*ABC::",
    components=["A", "B", "C"],
    interactions=["activates", "inhibits"],
    dt=20.0,  # seconds
    # nt = 2000,
    nt=200,
    batch_size=4,
    parameter_table=param_table,
    save_dir=save_dir,
    time_limit=5,  # for debugging
)

tree.search_mcts(10)

...
