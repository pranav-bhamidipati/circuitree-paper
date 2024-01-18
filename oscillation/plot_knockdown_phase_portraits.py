### Plot a PCA of the phase portraits for a wild-type oscillator and each of its knockdowns
# Steps
# =====
#
# Read in HDF file
#   Read in the time-series data of protein quantity with shape:
#       (n_runs, n_timesteps, n_proteins)
#   Compute the relative abundance of the proteins (i.e. normalize over axis -2)
#   Isolate wild-type oscillator data (knockdown_coeff == 1.0)
#   Find which species is peaking at each time-point by finding the argmax of
#       relative_abundance over axis -2 (which_peak_species)
#
# Compute PCA using the wild-type oscillator replicates.
#   PC1 and PC2 will be used to project the knockdown data onto a 2D phase portrait.
#
# To plot a phase portrait:
#   Project the knockdown data onto the PC1-PC2 plane
#   Use which_peak_species with a colormap to color each time-point
#   Plot the time-series with colors on the PC1-PC2 plane
#
# Plot a single replicate of the wild-type oscillator phase portrait
# Then, for each knockdown:
#   For each knockdown coefficient (in a given range):
#     Plot phase portrait for one replicate. Line style indicates which coefficient.
#
# Save

from datetime import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# import seaborn as sns
from pathlib import Path
import pandas as pd

from sklearn.decomposition import PCA


def remove_axis_and_ticks(ax=None):
    """Remove axis and ticks from a matplotlib axis while retaining axis labels."""

    ax = ax or plt.gca()

    # Remove x and y axis lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove ticks and tick labels
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def main(
    knockdown_data_hdf: Path,
    plot_coeffs: list[float],
    which_replicate: int = 0,
    species_cmap: str = "tab10",
    plot_all_replicates: bool = False,
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
):
    metadata = pd.read_hdf(knockdown_data_hdf, key="metadata")
    with h5py.File(str(knockdown_data_hdf), "r") as f:
        genotype = f.attrs["genotype"]
        y_ts = f["y_ts"][()]
        t_secs = f["t_secs"][()]

    # Isolate protein data
    components = genotype.strip("*").split("::")[0]
    n_components = len(components)
    prots_t = y_ts[..., n_components : n_components * 2]

    # Compute normalized quantities
    max_quantities = prots_t.max(axis=-2, keepdims=True)
    normalized_abundance = np.where(max_quantities == 0, 0.0, prots_t / max_quantities)
    which_most_abundant = normalized_abundance.argmax(axis=-1)

    # Compute PCA on the whole dataset
    pca = PCA(n_components=2)
    pca.fit(prots_t.reshape(-1, n_components))

    # Get indices of wild-type runs (where "knockdown_coeff" == 1.0)
    which_wt = np.where(metadata["knockdown_coeff"] == 1.0)[0]

    # Plot phase portrait of wild-type oscillator
    wt_prots_t = prots_t[which_wt[which_replicate]]
    wt_most_abundant = which_most_abundant[which_wt[which_replicate]]
    colors = colormaps.get_cmap(species_cmap)(wt_most_abundant)
    # cmap = sns.color_palette(species_cmap, n_colors=n_components, as_cmap=True)

    # fig, ax = plt.subplots(figsize=(4, 4))
    # xx, yy = pca.transform(wt_prots_t).T
    # # xx = xx[:500]
    # # yy = yy[:500]
    # # colors = colors[:500]
    # ax.scatter(xx, yy, s=1, c=colors)
    # ax.plot(xx, yy, c="k", lw=0.5, alpha=0.2)
    # ax.set_xlabel("PC1")
    # ax.set_ylabel("PC2")
    # ax.set_title("WT")

    # remove_axis_and_ticks()

    # if save:
    #     today = datetime.today().strftime("%y%m%d")
    #     fpath = (
    #         Path(save_dir) / f"{today}_knockdowns_wt_phase_portrait_color_guides.{fmt}"
    #     )
    #     print(f"Writing to: {fpath.resolve().absolute()}")
    #     plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(4, 4))
    xx, yy = pca.transform(wt_prots_t).T
    colors = colormaps.get_cmap(species_cmap)(wt_most_abundant)

    ax.scatter(xx, yy, s=50, c=colors, alpha=0.1)
    ax.plot(xx, yy, c="k", lw=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("WT")

    remove_axis_and_ticks()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir) / f"{today}_knockdowns_wt_phase_portrait.{fmt}"
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

    import seaborn as sns
    from matplotlib.colors import ListedColormap

    # Make an array of pca-transformed protein quantities for each knockdown
    # with shape (n_components, n_plot_coeffs, n_time_points, 2)
    # where the last axis is PC1 and PC2
    n_plot_coeffs = len(plot_coeffs)
    nt = len(t_secs)
    knockdown_pca = np.zeros((n_components, n_plot_coeffs, nt, 2))
    for which_kd, kd_component in enumerate(components):
        for which_coeff, coeff in enumerate(plot_coeffs):
            which_run = np.where(
                (metadata["knockdown_coeff"] == coeff)
                & (metadata["knockdown_tf"] == kd_component)
                & (metadata["replicate"] == which_replicate)
            )[0][0]
            kd_prots_t = prots_t[which_run]
            knockdown_pca[which_kd, which_coeff] = pca.transform(kd_prots_t)

    # Use turbo cmap with desaturated colors
    coeff_cmap = sns.color_palette("turbo", n_colors=n_plot_coeffs, desat=0.7)
    coeff_cmap = ListedColormap(coeff_cmap)
    xlim = knockdown_pca[..., 0].min(), knockdown_pca[..., 1].max()
    ylim = knockdown_pca[..., 1].min(), knockdown_pca[..., 1].max()
    for which_kd, kd_component in enumerate(components):
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()
        for which_coeff, coeff in enumerate(plot_coeffs):
            color = coeff_cmap(which_coeff / (n_plot_coeffs - 1))
            xx, yy = knockdown_pca[which_kd, which_coeff].T
            ax.plot(xx, yy, c=color, lw=1, label=f"{1-coeff:.0%}")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Knockdown: {kd_component}")
        ax.axis([*xlim, *ylim])

        # Place legend outside the plot, on the right side
        ax.legend(
            title="% Knockdown",
            frameon=False,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        remove_axis_and_ticks()

        if save:
            today = datetime.today().strftime("%y%m%d")
            fpath = (
                Path(save_dir)
                / f"{today}_knockdown_{kd_component}_phase_portrait.{fmt}"
            )
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

    ...


if __name__ == "__main__":
    kd_data_hdf = Path(
        "data/oscillation/240117_5TF_FTO3_knockdowns/240117_knockdowns_data.hdf5"
    )
    save_dir = Path("figures/oscillation/240116_FTO3_knockdowns")
    save_dir.mkdir(exist_ok=True)

    main(
        knockdown_data_hdf=kd_data_hdf,
        which_replicate=1,
        plot_coeffs=np.linspace(1.0, 0.0, 5),
        plot_all_replicates=True,
        save=True,
        save_dir=save_dir,
        # fmt="pdf",
    )
