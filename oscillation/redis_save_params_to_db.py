from itertools import islice
import json
import redis
import pandas as pd
from pathlib import Path

from oscillation_app import app


def _split_seq_into_chunks(seq, chunk_sizes):
    it = iter(seq)
    return [list(islice(it, size)) for size in chunk_sizes]


def main(
    param_sets_csv: str | Path,
    n_init_columns: int,
    n_parameters: int = 10,
    database_url: str | Path = None,
    index_column: str = "param_index",
    table_name: str = "parameter_table",
):
    print("Loading parameter sets")
    params_df = pd.read_csv(param_sets_csv, index_col=index_column)

    print("Connecting to database")
    database_url = database_url or app.conf["broker_url"]
    database: redis.Redis = redis.Redis.from_url(database_url)

    columns = params_df.columns
    ncol = len(columns)
    if ncol != n_init_columns + n_parameters:
        raise ValueError(
            f"Expected dataframe with {n_init_columns + n_parameters} columns, "
            f"not {ncol}."
        )
    params_df = params_df.sort_index()

    # JSON-serialize each row as a [seed, prots0, params] tuple
    chunk_sizes = (1, n_init_columns, n_parameters)
    params_mapping = {}
    for pidx, *row in params_df.itertuples():
        seed, prots0, params = _split_seq_into_chunks(row, chunk_sizes)
        params_mapping[str(pidx)] = json.dumps([seed, prots0, params])

    # Store in database
    clear_before_storing = (
        input("Delete parameter table in database before storing? [y/N] ").lower()
        == "y"
    )
    if clear_before_storing:
        print("Deleting parameter table...")
        _ = database.delete(table_name)
    else:
        print("Not clearing database.")
    print("Storing parameter sets in database...")
    _ = database.hset(table_name, mapping=params_mapping)

    print("Done.")


if __name__ == "__main__":
    param_sets_csv = Path(
        # "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000.csv"
        # "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000_5tf.csv"
        "~/git/circuitree-paper/data/oscillation/231020_param_sets_10000_5tf.csv"
    ).expanduser()

    main(
        param_sets_csv=param_sets_csv,
        n_init_columns=5,
        n_parameters=10,
    )
