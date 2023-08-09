from collections import Counter
from datetime import date
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


def analyze_mcts_results(
    rep_batch_size_and_csv,
    best100,
    osc_thresh=0.01,
):
    replicate, batch_size, csv = rep_batch_size_and_csv
    _df = pd.read_csv(csv)
    if "step" not in _df.columns:
        _df.index.name = "step"
        _df = _df.reset_index()
    _df["batch_size"] = batch_size
    _df["replicate"] = replicate
    _df = _df.dropna().sort_values("step")

    (
        n_oscillators,
        oscillators,
        num_visited,
    ) = compute_search_quality(
        _df["simulated_node"],
        _df["rewards"],
        osc_thresh=osc_thresh,
    )
    _df["n_oscillators"] = n_oscillators
    pct_top100 = [len(set(os) & best100) for os in oscillators]
    _df["pct_top100"] = pct_top100
    _df["num_visited"] = num_visited

    return _df


def compute_search_quality(states, rewards, osc_thresh=0.01):
    cum_visits = Counter()
    cum_reward = Counter()
    n_oscillators = np.zeros(len(states), dtype=int)
    oscillators = []
    num_visited = []
    osc = set()

    iterator = enumerate(zip(list(states), list(rewards)))
    for i, (s, r) in iterator:
        cum_visits[s] += 1
        cum_reward[s] += r
        s_reward = cum_reward[s]
        s_visits = cum_visits[s]
        p_osc = s_reward / s_visits

        if s.startswith("*"):
            if p_osc > osc_thresh:
                # print(f"Step {i} -- p_osc={p_osc:.4f} -- {s}")
                osc.add(s)
            else:
                osc.discard(s)

        if i % 1000 == 0:
            ...

        _s = "*ABC::AAa_ABa_ACi_BAi_BBa_CBi_CCa"
        if max(osc, key=lambda x: cum_reward[x] / cum_visits[x], default="") == _s:
            if i % 10 == 0:
                print(
                    f"Step {i} -- !!mixed osc!! -- p_osc={cum_reward[_s] / cum_visits[_s]:.4f}"
                )
            if i % 100 == 0:
                ...

        n_oscillators[i] = len(osc)

        oscillators.append(osc.copy())
        num_visited.append(len(cum_visits))

    return (
        n_oscillators,
        oscillators,
        num_visited,
    )


def main(
    mcts_data_dir: Path,
    results_csv: Path,
    nprocs: int = 1,
    osc_thresh: float = 0.01,
    save: bool = False,
    save_dir: Optional[Path] = None,
):
    results_df = pd.read_csv(results_csv, index_col=0)
    best_oscillators = results_df.sort_values("p_oscillation", ascending=False)
    best_oscillators = best_oscillators.head(100)["state"].reset_index(drop=True)
    best100 = set(best_oscillators)
    max_p_osc = results_df["p_oscillation"].max()

    mcts_data_dir = Path(mcts_data_dir)
    # csvs = list(mcts_data_dir.glob("*batchsize*.csv"))
    csvs = list(mcts_data_dir.glob("*batchsize10_*.csv"))
    batch_sizes = []
    for csv in csvs:
        batch_size = int(csv.stem.split("_")[3].split("batchsize")[1])
        batch_sizes.append(batch_size)
    replicates = pd.DataFrame(batch_sizes, columns=["b"]).groupby("b").cumcount().values

    args = zip(replicates, batch_sizes, csvs)
    analyze_in_parallel = partial(
        analyze_mcts_results,
        best100=best100,
        osc_thresh=osc_thresh,
    )

    pbar = tqdm(total=len(csvs), desc="Global progress", position=0)
    dfs = []
    if nprocs == 1:
        for arg in args:
            _df = analyze_in_parallel(arg)
            dfs.append(_df)
            pbar.update(1)
    else:
        with Pool(nprocs) as pool:
            for df in pool.imap_unordered(analyze_in_parallel, args):
                dfs.append(df)
                pbar.update(1)
    pbar.close()

    df = pd.concat(dfs, axis=0).set_index(["batch_size", "replicate", "step"])
    df.sort_index(inplace=True)
    df["regret"] = (
        (max_p_osc - df["rewards"]).groupby(["batch_size", "replicate"]).cumsum()
    )
    df.reset_index(inplace=True)

    if save:
        today = date.today().strftime("%y%m%d")
        fpath = (
            Path(save_dir).joinpath(f"{today}_search_summary.csv").resolve().absolute()
        )
        print(f"Writing to: {fpath}")
        df.to_csv(fpath, index=False)


if __name__ == "__main__":
    mcts_data_dir = Path("data/oscillation/230725_mcts_bootstrap_boolean2")
    results_csv = Path("data/oscillation/230717_motifs.csv")
    save_dir = Path("data/oscillation/230801_mcts_analysis")
    save_dir.mkdir(exist_ok=True)

    main(
        nprocs=1,
        # nprocs=None,
        mcts_data_dir=mcts_data_dir,
        results_csv=results_csv,
        # save=True,
        save_dir=save_dir,
    )
