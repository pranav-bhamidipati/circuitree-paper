from gevent import monkey

monkey.patch_all()

import datetime
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import redis
from tqdm import tqdm

from oscillation_app import app


def main(
    keyset="transposition_table_keys",
    save_dir: str | Path = None,
    prefix="",
    tz: datetime.timezone = None,
    progress_bar: bool = True,
    print_progress: bool = False,
    database_url: str | Path = None,
    logger: logging.Logger = None,
    dry_run: bool = False,
    **kwargs,
):
    stream_output = logger.info if logger is not None else print

    if print_progress and progress_bar:
        raise ValueError("print_progress and progress_bar cannot both be True.")

    # Connect to redis
    database_url = database_url or app.conf["broker_url"]
    stream_output(f"Connecting to redis at {database_url}")
    database = redis.Redis.from_url(database_url)

    # Collect all keys to back up
    keys_to_backup = database.smembers(keyset)
    if not keys_to_backup:
        stream_output(f"No keys found in keyset {keyset}. Backup aborted.")
        return

    # Convert redis data to parquet-formatted DataFrame
    _unpack = lambda k, vs: (int(k.decode()), *map(float, json.loads(vs.decode())))
    data = []

    if print_progress:
        print_points = np.linspace(0, len(keys_to_backup), 11)
        next_print_idx = 1
        next_print_point = print_points[next_print_idx]

    stream_output(f"Backing up {len(keys_to_backup)} keys...")

    if dry_run:
        stream_output("Option dry_run=True. Backup not performed.")
        return

    iterator = tqdm(keys_to_backup) if progress_bar else keys_to_backup
    for i, key in enumerate(iterator):
        param_indices, autocorr_mins, sim_times = zip(
            *(_unpack(k, v) for k, v in database.hgetall(key).items())
        )
        state_name = "*" + str(key.decode()).lstrip("state_")
        state_data = pd.DataFrame(
            {
                "state": state_name,
                "param_idx": param_indices,
                "autocorr_min": autocorr_mins,
                "sim_time": sim_times,
            }
        )
        data.append(state_data)

        if print_progress and i >= next_print_point:
            stream_output(f"{next_print_idx * 10}% complete.")
            next_print_idx += 1
            next_print_point = print_points[next_print_idx]

    df = pd.concat(data).sort_values(["state", "param_idx"])
    df["state"] = df["state"].astype("category")

    # Save to disk
    date_time_fmt = "%Y-%m-%d_%H-%M-%S"
    tz = tz or datetime.timezone.utc
    date_time_formatted = datetime.datetime.now(tz).strftime(date_time_fmt)
    save_dir = Path(save_dir) if save_dir else Path.cwd()
    filepath = (
        (save_dir / f"{prefix}backup_{date_time_formatted}.parquet")
        .resolve()
        .absolute()
    )
    stream_output(f"Saving backup to: {filepath}")
    df.to_parquet(filepath, index=False, engine="pyarrow")
    stream_output("Database backup complete.")


if __name__ == "__main__":
    save_dir = Path("~/git/circuitree-paper/data/oscillation/backups").expanduser()
    main(
        prefix="mcts_5tf_",
        save_dir=save_dir,
    )
