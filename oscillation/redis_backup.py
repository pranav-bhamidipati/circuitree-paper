### Convert redis data to parquet-formatted DataFrame and save to disk
### This script is intended to be run on a regular basis to backup the redis data

import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
import redis
from tqdm import tqdm


def main(
    keyset="transposition_table_keys",
    host="localhost",
    port=6379,
    db=0,
    save_dir: str | Path = None,
    prefix="",
    tz_offset=-7,  # Pacific Time
    progress_bar: bool = True,
    print_progress: bool = False,
    **kwargs,
):
    if print_progress and progress_bar:
        raise ValueError("print_progress and progress_bar cannot both be True.")

    # Connect to redis
    r = redis.Redis(host=host, port=port, db=db)

    # Collect all keys to back up
    keys_to_backup = r.smembers(keyset)

    # Convert redis data to parquet-formatted DataFrame
    _unpack = lambda k, vs: (int(k.decode()), *map(float, json.loads(vs.decode())))
    data = []

    if print_progress:
        print_points = np.linspace(0, len(keys_to_backup), 11)
        next_print_idx = 1
        next_print_point = print_points[next_print_idx]

    print(f"Backing up {len(keys_to_backup)} keys...")
    iterator = tqdm(keys_to_backup) if progress_bar else keys_to_backup
    for i, key in enumerate(iterator):
        visits, autocorr_mins, sim_times = zip(
            *(_unpack(k, v) for k, v in r.hgetall(key).items())
        )
        state_name = "*" + str(key.decode()).lstrip("state_")
        state_data = pd.DataFrame(
            {
                "state": state_name,
                "visit": visits,
                "autocorr_min": autocorr_mins,
                "sim_time": sim_times,
            }
        )
        data.append(state_data)

        if print_progress:
            if i >= next_print_point:
                print(f"{next_print_idx * 10}% complete.")
                next_print_idx += 1
                next_print_point = print_points[next_print_idx]

    df = pd.concat(data).sort_values(["state", "visit"])
    df["state"] = df["state"].astype("category")

    # Save to disk
    date_time_fmt = "%Y-%m-%d_%H-%M-%S"
    date_time_formatted = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=tz_offset))
    ).strftime(date_time_fmt)
    save_dir = Path(save_dir) if save_dir else Path.cwd()
    filepath = (
        (save_dir / f"{prefix}backup_{date_time_formatted}.parquet")
        .resolve()
        .absolute()
    )
    print(f"Saving backup to: {filepath}")
    df.to_parquet(filepath, index=False)
    print("Database backup complete.")


if __name__ == "__main__":
    save_dir = Path("~/git/circuitree-paper/data/oscillation/backups").expanduser()
    main(
        prefix="mcts_5tf_",
        save_dir=save_dir,
    )
