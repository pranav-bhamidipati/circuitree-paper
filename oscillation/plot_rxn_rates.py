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
    nt_sampled: int = 360,
    nt_total: int = 2_000,
    save: bool = False,
    save_dir: Path = None,
    figsize: tuple[float, float] = (5, 3.5),
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

    df["runtime_secs"] = df["runtime_secs"] / nt_sampled * nt_total
    df["log_runtime_secs"] = np.log10(df["runtime_secs"])
    df["reactions_per_min"] = df["iterations_per_timestep"] * 60.0 / dt_seconds
    df["reactions_per_min"] = np.maximum(df["reactions_per_min"], 10**nan_val)
    df["log_reactions_per_min"] = np.log10(df["reactions_per_min"])
    # df["log_reactions_per_min"] = df["log_reactions_per_min"].replace(-np.inf, nan_val)

    reaction_quantiles = df["reactions_per_min"].quantile(quantiles)
    q_labels = [f"{100 * q:.2f}".rstrip("0").rstrip(".") + "%" for q in quantiles]
    q_colors = sns.color_palette("pastel", n_colors=len(quantiles))

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=df, x="reactions_per_min", log_scale=True, ax=ax)
    plt.xlabel(r"Reaction rate (min$^{-1}$)")
    ymin, ymax = plt.ylim()
    line_kw = dict(
        ymin=ymin,
        ymax=ymax,
        # color="gray",
        linestyles="dashed",
        lw=1,
    )
    for q, q_label, q_color in zip(reaction_quantiles, q_labels, q_colors):
        plt.vlines(q, label=q_label, color=q_color, **line_kw)
    plt.legend(title="Percentile")
    plt.ylim(ymin, ymax)
    sns.despine()
    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"reaction_rates_hist.{fmt}").resolve().absolute()
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=df, x="reactions_per_min", y="runtime_secs", log_scale=(True, True), ax=ax
    )
    plt.xlabel(r"Reaction rate (min$^{-1}$)")
    plt.ylabel(r"Simulation time (s)")
    sns.despine()
    plt.tight_layout()

    if save:
        fname = (
            save_dir.joinpath(f"runtime_vs_reaction_rate.{fmt}").resolve().absolute()
        )
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig, ax = plt.subplots(figsize=figsize)
    sns.ecdfplot(
        data=df, x="reactions_per_min", hue="complexity", log_scale=True, ax=ax
    )
    plt.xlabel(r"Reaction rate (min$^{-1}$)")
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
    sns.ecdfplot(data=df, x="runtime_secs", hue="complexity", log_scale=True, ax=ax)
    plt.xlabel(r"Simulation time (s)")
    sns.despine()
    plt.tight_layout()

    if save:
        fname = (
            save_dir.joinpath(f"runtime_ecdf_v_complexity.{fmt}").resolve().absolute()
        )
        print(f"Saving figure to: {fname}")
        plt.savefig(fname, dpi=dpi)

    fig = plt.figure(figsize=(figsize[0] * 2, figsize[1] * 2))
    sampled_params = {
        "log10_kd_1": r"$\log_{10}K_d^{(1)}$",
        "kd_2_1_ratio": r"$K_d^{(2)}/K_d^{(1)}$",
        "km_unbound": r"$k_0$",
        "km_act": r"$k_a$",
        "nlog10_km_rep_unbound_ratio": r"$-\log_{10}(k_r/k_0)$",
        "kp": r"$k_p$",
        "gamma_m": r"$\gamma_m$",
        "gamma_p": r"$\gamma_p$",
    }
    for i, (param_name, param_repr) in enumerate(sampled_params.items()):
        ax = fig.add_subplot(3, 3, i + 1)
        sns.histplot(
            data=df,
            x=param_name,
            y="reactions_per_min",
            ax=ax,
            log_scale=(False, True),
        )
        ax.set_xlabel(param_repr)
        ax.set_ylabel(r"Reaction rate (min$^{-1}$)")
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
        s=1,
        linewidth=0,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel(sampled_params["gamma_m"])
    ax.set_ylabel(sampled_params["log10_kd_1"])
    norm = plt.Normalize(
        df["log_reactions_per_min"].min(), df["log_reactions_per_min"].max()
    )
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    fig.colorbar(sm, ax=ax, label=r"$\log_{10}$(reaction rate (min$^{-1}$)")
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
    plt.title(r"Top 1% of reaction rates")
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
    ax.set_xlabel(sampled_params["gamma_m"])
    ax.set_ylabel(sampled_params["log10_kd_1"])
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
    data_dir = Path("data/oscillation/230808_rxn_rate")
    save_dir = Path("figures/oscillation/reaction_rates")
    save_dir.mkdir(exist_ok=True)
    main(
        data_dir=data_dir,
        save_dir=save_dir,
        save=True,
    )
