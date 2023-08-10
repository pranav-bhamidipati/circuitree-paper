from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import pandas as pd
from pathlib import Path
import seaborn as sns
from tqdm import tqdm


def _draw_bootstrap_indices(
    arr_size: np.ndarray,
    batch_size: int,
    n_bootstraps: int,
):
    """Draw bootstrap samples of indices into a 1D array."""
    return np.random.randint(0, arr_size, size=(n_bootstraps, batch_size))


def _compute_batch_runtime_metrics(
    runtimes: np.ndarray,
    reaction_rates: np.ndarray,
    bootstrap_indices: np.ndarray,
    rate_threshold: float,
):
    # Get samples where the reaction rate is above the threshold
    mask = reaction_rates > rate_threshold

    # Compute the runtime of each batch, applying the threshold
    runtimes_masked = ma.masked_array(runtimes, mask=mask)
    batch_runtimes = runtimes_masked[bootstrap_indices].max(axis=1).compressed()

    q5, q95 = np.nanquantile(batch_runtimes, q=(0.05, 0.95))
    return {
        "rate_threshold": rate_threshold,
        "mean": np.nanmean(batch_runtimes),
        "std": np.nanstd(batch_runtimes),
        "median": np.nanmedian(batch_runtimes),
        "min": np.nanmin(batch_runtimes),
        "max": np.nanmax(batch_runtimes),
        "q5": q5,
        "q95": q95,
    }


def compute_runtime_metrics(
    runtimes: np.ndarray,
    reaction_rates: np.ndarray,
    batch_size: int,
    runtime_thresholds: tuple[float, ...],
    n_bootstraps: int,
):
    """Compute bootstrap metrics from a 1D array."""
    n = runtimes.size
    bs_indices = _draw_bootstrap_indices(n, batch_size, n_bootstraps)
    data = []
    for rt in runtime_thresholds:
        metrics = _compute_batch_runtime_metrics(
            runtimes, reaction_rates, bs_indices, rt
        )
        metrics["batch_size"] = batch_size
        data.append(metrics)
    return pd.DataFrame(data)


def main(
    data_dir: Path,
    n_bootstraps: int = 1_000_000,
    batch_sizes: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
    threshold_lim: tuple[float, ...] = (700_000, 10_000),
    n_thresholds: int = 21,
    dt_seconds: float = 20.0,
    nan_val: float = 0.0,
    figsize: tuple[float, float] = (9, 6),
    plot_shape: tuple[int, int] = (2, 3),
    save: bool = True,
    save_dir: Path | None = None,
    fmt: str = "png",
    dpi: int = 300,
):
    pqs = list(data_dir.glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(pq) for pq in tqdm(pqs, desc="Loading data")])

    df.index.name = "batch_idx"
    df = df.reset_index().set_index(["state", "batch_idx"]).sort_index().reset_index()

    # Encode circuit "complexity" (number of interactions), accounting for the null case
    df["complexity"] = 1 + df.state.str.count("_")
    df["complexity"] = df["complexity"].where(df["state"] != "*ABC::", 0)
    df["complexity"] = pd.Categorical(df["complexity"], ordered=True)

    df["log_runtime_secs"] = np.log10(df["runtime_secs"])
    df["reactions_per_min"] = df["iterations_per_timestep"] * 60.0 / dt_seconds
    df["log_reactions_per_min"] = np.log10(df["reactions_per_min"])
    df["log_reactions_per_min"] = df["log_reactions_per_min"].replace(-np.inf, nan_val)

    # Draw bootstrap samples of runtime for each batch size
    runtimes = df["runtime_secs"].values * 5.5
    reaction_rates = df["reactions_per_min"].values
    thresholds = np.geomspace(*threshold_lim, n_thresholds)

    # Compute the fraction of samples excluded by the threshold
    p_excluded = np.array([df["reactions_per_min"].gt(t).mean() for t in thresholds])

    print("Plotting...")

    fig = plt.figure(figsize=(figsize[0] / 2, figsize[1] / 2))
    ax = fig.add_subplot(1, 1, 1)
    p_excluded_data = pd.DataFrame(
        {
            "rate_threshold": thresholds,
            "pct_excluded": p_excluded,
        }
    )
    sns.lineplot(
        data=p_excluded_data,
        x="rate_threshold",
        y="pct_excluded",
        ax=ax,
        errorbar=None,
    )
    plt.hlines(0.01, *threshold_lim, linestyles="--", colors="gray", label="1%")
    plt.legend()

    ax.set_xlabel(r"Reaction rate cutoff (min$^{-1}$)")
    ax.set_xscale("log")
    ax.set_xlim(threshold_lim)
    ax.set_ylabel("Fraction of samples excluded")
    plt.tight_layout()

    if save:
        save_dir = Path(save_dir)
        fpath = (
            save_dir.joinpath(f"pct_excluded_vs_rate_threshold.{fmt}")
            .resolve()
            .absolute()
        )
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi)

    compute_metrics_for_batchsize = partial(
        compute_runtime_metrics,
        runtimes=runtimes,
        reaction_rates=reaction_rates,
        runtime_thresholds=thresholds,
        n_bootstraps=n_bootstraps,
    )
    dfs = []
    for i, batch_size in enumerate(batch_sizes):
        print(f"Computing bootstrap data for batch size {batch_size}...")
        _df = compute_metrics_for_batchsize(batch_size=batch_size)
        dfs.append(_df)
    df_bs = pd.concat(dfs)
    df_bs = df_bs.sort_values(["batch_size", "rate_threshold"], ascending=[True, False])

    fig = plt.figure(figsize=figsize)
    for i, batch_size in enumerate(batch_sizes):
        batch_data = df_bs.loc[df_bs["batch_size"] == batch_size]
        ax = fig.add_subplot(*plot_shape, i + 1)
        opts = dict(data=batch_data, x="rate_threshold", ax=ax, errorbar=None)

        ax.set_title(f"Batch size: {batch_size}")
        ax.set_xscale("log")
        ax.set_xlim(threshold_lim)
        ax.set_xlabel(r"Reaction rate cutoff (min$^{-1}$)")
        ax.set_ylabel("Batch runtime (s)")
        ax.set_yscale("log")
        ax.set_ylim(3.0e-1, 1.1e3)

        blues = sns.color_palette("Blues", 4)

        plt.fill_between(
            batch_data["rate_threshold"],
            batch_data["min"],
            batch_data["max"],
            color=blues[0],
        )
        plt.fill_between(
            batch_data["rate_threshold"],
            batch_data["q5"],
            batch_data["q95"],
            color=blues[1],
        )
        sns.lineplot(y="mean", color=blues[3], **opts)

        # sns.lineplot(y="q5", label="q5", **opts)
        # sns.lineplot(y="mean", label="mean", **opts)
        # sns.lineplot(y="q95", label="q95", **opts)
        # sns.lineplot(y="max", label="max", **opts)

    plt.tight_layout()

    if save:
        save_dir = Path(save_dir)
        fpath = (
            save_dir.joinpath(f"runtime_with_rxn_rate_threshold.{fmt}")
            .resolve()
            .absolute()
        )
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    data_dir = Path("data/oscillation/230808_rxn_rate")
    save_dir = Path("figures/oscillation/230808_rxn_rate")
    save_dir.mkdir(exist_ok=True)
    main(
        data_dir=data_dir,
        save_dir=save_dir,
    )
