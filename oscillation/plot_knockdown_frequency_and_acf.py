### Make 3 plots that study oscillator quality vs. % knockdown, for each TF
### 1. A histogram of oscillating samples for each knockdown vs. % knockdown
### 2. Line plot of ACF minima vs. % knockdown
### 3. Scatter plot with line and error bars of oscillatiion frequency vs. % knockdown

from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings


def main(
    knockdown_data_hdf: Path,
    acf_threshold: float = -0.4,
    species_cmap: str = "tab10",
    figsize: tuple[float, float] = (4, 2),
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
):
    warnings.filterwarnings("ignore", category=FutureWarning)

    metadata = pd.read_hdf(knockdown_data_hdf, key="metadata")
    with h5py.File(str(knockdown_data_hdf), "r") as f:
        acf_minima = f["acf_minima"][()]
        frequencies = f["frequencies"][()]
    metadata["pct_knockdown"] = 100 * (1 - metadata["knockdown_coeff"])
    metadata["pct_knockdown"] = metadata["pct_knockdown"].astype(int)
    metadata["acf_min"] = acf_minima
    metadata["freq_per_hour"] = frequencies * 3600

    # Plot ACF minima vs. % knockdown, with an envelope of the 95% confidence interval
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        x="pct_knockdown",
        y="acf_min",
        hue="knockdown_tf",
        data=metadata,
        ax=ax,
        errorbar=("ci", 95),
        palette=species_cmap,
        legend=False,
    )

    # Dashed horizontal line at ACF threshold for oscillation
    ax.axhline(acf_threshold, linestyle="--", color="gray", alpha=0.65)

    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_ylabel(r"ACF$_\mathrm{min}$")
    sns.despine()
    plt.tight_layout()

    if save:
        fpath = Path(save_dir).joinpath(f"acf_minima_vs_pct_KD.{fmt}")
        print(f"Saving to {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    # Plot oscillation frequency vs. % knockdown
    # Use markers with error bars + lines
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(
        species_cmap, n_colors=metadata["knockdown_tf"].nunique()
    )
    osc_mask = metadata["acf_min"] < acf_threshold
    for tf, color in zip(metadata["knockdown_tf"].unique()[::-1], colors[::-1]):
        tf_mask = (metadata["knockdown_tf"] == tf) & osc_mask
        grouped = metadata.loc[tf_mask].groupby("pct_knockdown")["freq_per_hour"]
        mean_freq = grouped.mean()
        pct_kd = mean_freq.index
        mean_freq = mean_freq.values
        ci_lo = grouped.quantile(0.025).values
        ci_hi = grouped.quantile(0.975).values
        yerr_lo = mean_freq - ci_lo
        yerr_hi = ci_hi - mean_freq

        # Force nonnegative (negatives happen due to floating point errors)
        yerr_lo[yerr_lo < 0] = 0
        yerr_hi[yerr_hi < 0] = 0

        # 95% confidence interval
        ax.errorbar(
            pct_kd,
            mean_freq,
            yerr=(yerr_lo, yerr_hi),
            label=tf,
            color=color,
            fmt="o",
            markersize=5,
            capsize=2,
        )
        plt.plot(pct_kd, mean_freq, color=color)

    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_ylabel(r"Frequency (hour$^{-1}$)")
    sns.despine()
    plt.tight_layout()

    if save:
        fpath = Path(save_dir).joinpath(f"frequency_vs_pct_KD.{fmt}")
        print(f"Saving to {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    # Plot histogram of oscillating samples for each knockdown vs. % knockdown
    fig, axs = plt.subplots(5, 1, figsize=(figsize[0], figsize[1] * 0.5))
    n_bins = metadata["pct_knockdown"].nunique()
    bw = 100 / (n_bins - 1)
    bin_edges = np.linspace(-bw / 2, 100 + bw / 2, n_bins + 1)
    tfs = metadata["knockdown_tf"].unique()
    for ax, tf, color in zip(axs, tfs, colors):
        tf_mask = (metadata["acf_min"] < acf_threshold) & (
            metadata["knockdown_tf"] == tf
        )
        sns.histplot(
            data=metadata.loc[tf_mask],
            x="pct_knockdown",
            multiple="stack",
            color=color,
            bins=bin_edges,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.set_ylabel("")
        ax.set_yticks([metadata["replicate"].nunique()])
        if tf != tfs[-1]:
            ax.set_xticks([])
        sns.despine()

    if save:
        fpath = Path(save_dir).joinpath(f"hist_samples_vs_pct_KD.{fmt}")
        print(f"Saving to {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":

    from datetime import datetime

    today = datetime.today().strftime("%y%m%d")
    save_dir = Path(f"figures/oscillation/{today}_FTO3_knockdowns")
    save_dir.mkdir(exist_ok=True)

    kd_data_hdf = Path(
        "data/oscillation/240118_5TF_FTO3_knockdowns/240119_knockdowns_data.hdf5"
    )
    main(
        knockdown_data_hdf=kd_data_hdf,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
