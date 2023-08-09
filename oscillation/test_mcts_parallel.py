from typing import Optional
from oscillation.oscillation_parallel import *
from multiprocessing import Pool
from pathlib import Path
from circuitree.parallel import TranspositionTable


def main(
    data_loc: Optional[Path] = None,
    init_columns: Optional[list[str]] = None,
    param_names: Optional[list[str]] = None,
    components: Optional[list[str]] = None,
    interactions: Optional[list[str]] = None,
    root: str = "ABC::",
    reward_column: str = "reward",
    load_kw: dict = {},
):
    if init_columns is None:
        init_columns = ["A_0", "B_0", "C_0"]
    if param_names is None:
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

    if components is None:
        components = ["A", "B", "C"]
    if interactions is None:
        interactions = ["activates", "inhibits"]

    columns = [reward_column] + init_columns + param_names
    kw = dict(
        columns=columns,
        nt=100,
        dt=20.0,
        components=components,
        interactions=interactions,
        root=root,
        bootstrap=True,
    )

    # Read a transposition table from a file (or multiple files)
    data_path = Path(data_loc)
    if data_path.is_dir():
        pqfiles = list(data_path.glob("state_*/*.parquet"))
        if not pqfiles:
            raise ValueError(f"No parquet files found in data directory: {data_path}")
        table = TranspositionTable.from_parquet(
            pqfiles,
            init_columns=init_columns,
            param_columns=param_names,
            progress=True,
            **load_kw,
        )
    elif data_path.exists():
        if data_path.suffix != ".parquet":
            raise ValueError(f"Data file has invalid extension: {data_path}")
        table = TranspositionTable.from_parquet(
            data_path,
            init_columns=init_columns,
            param_columns=param_names,
            progress=True,
            **load_kw,
        )

    ...

    with Pool() as p:
        ot = OscillationTreeParallel(p, transposition_table=table, read_only=True, **kw)
        ot.batch_size = 192
        total_samples = 34_110_000
        n_batch_samples = total_samples // ot.batch_size
        ot.search_mcts(n_batch_samples, progress_bar=True)
        ...


if __name__ == "__main__":
    # data_loc = Path("data/oscillation/bfs_230710_hpc")
    data_loc = Path("data/oscillation/230713_transposition_table_hpc.parquet")
    main(data_loc=data_loc, load_kw=dict(index=False))
