from circuitree.viz import plot_network
from datetime import date
import h5py
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from circuitree.models import SimpleNetworkGrammar
from tf_network import (
    autocorrelate,
    compute_lowest_minimum,
    binomial9_kernel,
)


grammar = SimpleNetworkGrammar(
    components=["A", "B", "C"], interactions=["activates", "inhibits"]
)


def has_motif(state: str, motif: str):
    interaction_code = state.split("::")[1]
    if not interaction_code:
        return False
    state_interactions = set(interaction_code.split("_"))

    for recoloring in grammar.get_interaction_recolorings(motif):
        motif_interactions = set(recoloring.split("_"))
        if motif_interactions.issubset(state_interactions):
            return True
    return False


def plot_network_quantity_and_acorr(
    genotype,
    t,
    y_t,
    acorr,
    corr_time,
    minimum_val,
    plot_dir,
    save,
    dpi,
    fmt,
    figsize=(2.5, 2.5),
    # plot_motifs=(),
    suptitle=None,
    suffix="",
    **kwargs,
):
    n_species, nt = y_t.shape
    half_nt = nt // 2
    t_mins = t / 60.0

    fig1, ax = plt.subplots(1, 1, figsize=(figsize[0] * 0.6, figsize[1] * 0.6))
    plt.sca(ax)
    plot_network(
        *grammar.parse_genotype(genotype, nonterminal_ok=True), ax=ax, **kwargs
    )
    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_network_diagram{suffix}.{fmt}"
        fpath = plot_dir.joinpath(fname).resolve().absolute()
        print("Writing to:", fpath)
        plt.savefig(fpath, dpi=dpi, transparent=True)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    plt.sca(ax1)
    for j in range(n_species):
        ax1.plot(t_mins, y_t[j], lw=1)

    # if plot_motifs:
    #     x = 0.98
    #     y = 0.95
    #     ax.text(
    #         x, y, "Motifs:", ha="left", va="top", transform=ax.transAxes, size=8
    #     )
    #     y -= 0.1
    #     for motif, s in plot_motifs:
    #         if has_motif(state, motif):
    #             ax.text(x, y, s, ha="left", va="top", transform=ax.transAxes, size=8)
    #             y -= 0.1

    ax1.set_ylabel("TF Quantity")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    plt.locator_params(axis="y", tight=True, nbins=3)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.sca(ax2)
    for j in range(n_species):
        ax2.plot(t_mins[:-half_nt], acorr[j, :], lw=1)
    corr_time_mins = corr_time / 60.0
    # ax3.scatter(corr_time_mins, minimum_val, marker="x", s=50, c="r", zorder=100)
    ax2.annotate(
        rf"$\mathrm{{ACF}}_\mathrm{{min}}={{{minimum_val:.2f}}}$",
        (corr_time_mins, minimum_val),
        (corr_time_mins + 0.1 * t_mins.max(), -1.3),
        arrowprops=dict(arrowstyle="->"),
        # ha="left",
        # va="bottom",
        size=10,
    )
    ax2.set_xlabel("Time (mins)")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_ylim(-1.5, 1.0)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    if suptitle is not None:
        plt.suptitle(suptitle, size=14)
    plt.tight_layout()

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_copynum_and_acf_example{suffix}.{fmt}"
        fpath = Path(plot_dir).joinpath(fname).resolve().absolute()
        print("Writing to:", fpath)
        plt.savefig(fpath, dpi=dpi, transparent=True)


def main(
    genotype,
    data_fpath,
    plot_dir,
    which_plot=0,
    save=False,
    dpi=300,
    fmt="png",
    suffix="",
    network_kwargs={},
):
    with h5py.File(data_fpath, "r") as f:
        t = f["t"][...]
        y_t = f["y_t"][which_plot]
        state = f["states"][0]

    # plot_motifs = [
    #     ("*ABC::ABa_BAi", "A-I"),
    #     ("*ABC::ABi_BCi_CAi", "I-I-I"),
    #     ("*ABC::ABi_BAi", "Toggle"),
    # ]

    # Before autocorrelation, we filter the data with a 9-point binomial filter
    filtered9 = np.apply_along_axis(binomial9_kernel, -1, y_t)[:, 4:-4]
    acorrs_f9 = np.apply_along_axis(autocorrelate, -1, filtered9)
    t_f9 = t[4:-4]
    all_minima_idx, all_minima = zip(*[compute_lowest_minimum(a) for a in acorrs_f9])
    which_lowest = np.argmin(all_minima)
    where_minimum = all_minima_idx[which_lowest]
    minimum_val = all_minima[which_lowest]
    corr_time = t_f9[where_minimum]

    if not suffix:
        gen_str = genotype.split(":")[-1]
        suffix = f"_{gen_str}_{which_plot}"

    plot_network_quantity_and_acorr(
        # suptitle=r"Highest reward - Copy number vs $t$ (mins)",
        genotype=genotype,
        t=t_f9,
        y_t=filtered9,
        acorr=acorrs_f9,
        corr_time=corr_time,
        minimum_val=minimum_val,
        plot_dir=plot_dir,
        save=save,
        dpi=dpi,
        fmt=fmt,
        # plot_motifs=plot_motifs,
        suffix=suffix,
        **network_kwargs,
    )

    ...


if __name__ == "__main__":
    data_fpath = Path(
        "data/oscillation/bfs_230710_hpc/top_oscillating_runs_ABC::AAa_ABa_BAi_CBi.hdf5"
    )
    # data_fpath = Path("data/oscillation/bfs_230710_hpc/top_oscillating_states.hdf5")
    # data_fpath = Path("data/oscillation/bfs_230710_hpc/top_oscillating_runs.hdf5")
    plot_dir = Path("figures/oscillation")

    network_kwargs = dict(
        fontsize=12,
        padding=0.5,
        lw=1,
        node_shrink=0.7,
        offset=0.8,
        auto_shrink=0.9,
        width=0.005,
    )

    main(
        genotype="ABC::AAa_ABa_BAi_CBi",
        data_fpath=data_fpath,
        plot_dir=plot_dir,
        which_plot=0,
        network_kwargs=network_kwargs,
        save=True,
        fmt="eps",
    )
