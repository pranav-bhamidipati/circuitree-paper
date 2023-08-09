import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from tqdm import tqdm


def main(
    data_dir: Path,
    nan_val: float = 0.0,
    dt_seconds: float = 20.0,
    quantiles: tuple = (0.95, 0.99, 0.999, 0.9999),
    save: bool = False,
    save_dir: Path = None,
    figsize: tuple[float, float] = (4, 3),
    fmt: str = "png",
    dpi: int = 300,
):
    if save and save_dir is None:
        raise ValueError("Must specify save_dir if save is True.")
    elif save:
        save_dir = Path(save_dir)

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
    df["log_reactions_per_min"] = np.log10(df["iterations_per_timestep"])
    df["log_reactions_per_min"] = df["log_reactions_per_min"].replace(-np.inf, nan_val)

    # To plot:
    # - [done] runtime vs reaction rates
    # - [done] effect of complexity (ECDFs)
    # - scatterplot reaction rates vs parameters
    # - [corner plots for any important parameters]
    # - Expected runtime vs batch size
    # - Expected runtime vs batch size with reaction rate cutoff
    #    - quantiles - 0.95, 0.99, 0.999, 0.9999
    #    - 50_000 reactions per minute

    reaction_quantiles = df["log_reactions_per_min"].quantile(quantiles)
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=df, x="log_reactions_per_min", ax=ax)
    plt.xlabel(r"$\log_{10}$(reactions/min)")
    ymin, ymax = plt.ylim()
    line_kw = dict(ymin=ymin, ymax=ymax, color="gray", linestyles="dashed", lw=1)
    for q in reaction_quantiles:
        plt.vlines(q, **line_kw)
    plt.ylim(ymin, ymax)
    sns.despine()
    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"reaction_rates_hist.{fmt}").resolve().absolute()
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig, ax = plt.subplots(figsize=figsize)
    sns.ecdfplot(data=df, x="log_reactions_per_min", hue="complexity", ax=ax)
    plt.xlabel(r"$\log_{10}$(reactions/min)")
    sns.despine()
    plt.tight_layout()

    if save:
        fname = (
            save_dir.joinpath(f"reaction_rate_ecdf_v_complexity.{fmt}")
            .resolve()
            .absolute()
        )
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig, ax = plt.subplots(figsize=figsize)
    sns.ecdfplot(data=df, x="log_runtime_secs", hue="complexity", ax=ax)
    plt.xlabel(r"$\log_{10}$(runtime) (seconds)")
    sns.despine()
    plt.tight_layout()

    if save:
        fname = (
            save_dir.joinpath(f"runtime_ecdf_v_complexity.{fmt}").resolve().absolute()
        )
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig = plt.figure(figsize=(figsize[0] * 3, figsize[1] * 3))
    sampled_param_names = [
        "log10_kd_1",
        "kd_2_1_ratio",
        "km_unbound",
        "km_act",
        "nlog10_km_rep_unbound_ratio",
        "kp",
        "gamma_m",
        "gamma_p",
    ]
    for i, param_name in enumerate(sampled_param_names):
        ax = fig.add_subplot(3, 3, i + 1)
        sns.histplot(
            data=df,
            x="log_reactions_per_min",
            y=param_name,
            ax=ax,
        )
        ax.set_xlabel(r"$\log_{10}$(reactions/min)")
        ax.set_ylabel(param_name)
        sns.despine()
    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"reaction_rate_v_params.{fmt}").resolve().absolute()
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    palette = "viridis"
    sns.scatterplot(
        data=df,
        x="gamma_m",
        y="log10_kd_1",
        hue="log_reactions_per_min",
        s=2,
        linewidth=0,
        palette=palette,
        ax=ax,
    )
    norm = plt.Normalize(
        df["log_reactions_per_min"].min(), df["log_reactions_per_min"].max()
    )
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    fig.colorbar(sm, ax=ax)
    sns.despine()
    plt.tight_layout()

    if save:
        fname = (
            save_dir.joinpath(f"reaction_rate_w_log10_kd_1_v_gamma_m.{fmt}")
            .resolve()
            .absolute()
        )
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    q99 = df["log_reactions_per_min"].quantile(0.99)
    top_1pct = (df["log_reactions_per_min"] > q99).values
    hi_color = plt.get_cmap("Dark2")(0)
    lo_color = plt.get_cmap("gray")(0.5)
    colors = [hi_color if p else lo_color for p in top_1pct]
    alphas = np.array([0.1, 1.0])[top_1pct.astype(int)]
    sizes = np.array([2, 4])[top_1pct.astype(int)]
    sns.scatterplot(
        data=df,
        x="gamma_m",
        y="log10_kd_1",
        s=sizes,
        c=colors,
        alpha=alphas,
        linewidth=0,
        ax=ax,
    )

    # Remove the legend
    sns.despine()
    plt.tight_layout()

    if save:
        fname = (
            save_dir.joinpath(f"reaction_rate_top1pct_w_log10_kd_1_v_gamma_m.{fmt}")
            .resolve()
            .absolute()
        )
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)


if __name__ == "__main__":
    data_dir = Path("data/oscillation/n_iter")
    save_dir = Path("figures/oscillation/n_iter")
    save_dir.mkdir(exist_ok=True)
    main(
        data_dir=data_dir,
        save_dir=save_dir,
        save=True,
    )
