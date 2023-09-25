### From a parquet file, we can recover the data and re-insert it into redis.
### Intended to reverse the process of redis_backup.py.


import json
import pandas as pd
from pathlib import Path
import redis
from tqdm import tqdm

from oscillation_app import app


def main(
    backup_file_pq: str | Path,
    restore_method="partial",
    column_names=["state", "visit", "autocorr_min", "sim_time"],
    state_keyset="transposition_table_keys",
    state_key_prefix="state_",
    prompt_on_delete=True,
    database_url: str | Path = None,
):
    # Determine restore method
    if restore_method == "full":
        delete_existing = True
        replace_if_exists = True
    elif restore_method == "partial":
        delete_existing = False
        replace_if_exists = False
    elif restore_method == "update":
        delete_existing = False
        replace_if_exists = True
    else:
        raise ValueError(
            f"Invalid restore_method: {restore_method}. "
            "Must be one of 'full', 'partial', or 'update'."
        )

    # Connect to redis
    database_url = database_url or app.conf["broker_url"]
    print(f"Connecting to redis at {database_url}")
    r = redis.Redis.from_url(database_url)

    # Load parquet file into DataFrame
    df = pd.read_parquet(backup_file_pq, columns=column_names)

    # Delete existing keys and their contents
    if delete_existing:
        if prompt_on_delete:
            print(f"Are you sure you want to delete all keys in keyset {state_keyset}?")
            response = input("  [y/N]: ")
            if response.lower() != "y":
                print("Exiting.")
                return

        print(f"Deleting existing keys in keyset: {state_keyset}")
        for key in r.smembers(state_keyset):
            r.delete(key)
        print(f"Deleting keyset: {state_keyset}")
        r.delete(state_keyset)

    # Insert data into redis
    print(f"Inserting data into redis with {replace_if_exists=}...")
    hset_func = r.hset if replace_if_exists else r.hsetnx
    for state, state_df in tqdm(df.groupby("state"), total=df["state"].nunique()):
        state_key = state_key_prefix + state.strip("*")
        r.sadd(state_keyset, state_key)
        for visit, autocorr_min, sim_time in zip(
            state_df["visit"], state_df["autocorr_min"], state_df["sim_time"]
        ):
            hset_func(state_key, visit, json.dumps([autocorr_min, sim_time]))


if __name__ == "__main__":
    backup_dir = Path("~/git/circuitree-paper/data/oscillation/backups").expanduser()
    # backup_file = backup_dir / "mcts_3tf_backup_2023-09-13_16-46-46.parquet"
    backup_file = backup_dir / "mcts_5tf_backup_2023-09-25_11-21-54.parquet"
    main(
        backup_file_pq=backup_file,
        restore_method="partial",
    )
