import json
import redis
import pandas as pd
from pathlib import Path

from oscillation_app import app


def _split_seq_at_index(seq, i):
    return seq[:i], seq[i:]


def main(
    param_sets_csv: str | Path,
    n_init_columns: int,
    n_parameters: int = 10,
    database_url: str | Path = None,
    seed_column: str = "sample_num",
    table_name: str = "parameter_table",
):
    print("Loading parameter sets")
    params_df = pd.read_csv(param_sets_csv, index_col=seed_column)

    print("Connecting to database")
    database_url = database_url or app.conf["broker_url"]
    database = redis.Redis.from_url(database_url)

    columns = params_df.columns
    ncol = len(columns)
    if ncol != n_init_columns + n_parameters:
        raise ValueError(
            f"Expected dataframe with {n_init_columns + n_parameters} columns, "
            f"not {ncol}."
        )
    params_df = params_df.sort_index()

    # JSON-serialize each row as a [prots0, params] pair
    params_mapping = {}
    for seed, *row in params_df.itertuples():
        prots0, params = _split_seq_at_index(row, n_init_columns)
        params_mapping[str(seed)] = json.dumps([prots0, params])

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
        "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000_5tf.csv"
    ).expanduser()

    main(
        param_sets_csv=param_sets_csv,
        n_init_columns=5,
        n_parameters=10,
        seed_column="sample_num",
    )
