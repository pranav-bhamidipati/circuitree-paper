from circuitree import SimpleNetworkGrammar
from circuitree.viz import plot_network
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
import numpy as np
from typing import Optional


_network_kwargs = dict(
    fontsize=6,
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
    plot_labels=False,
)


def main(
    data_hdf: Path,
    components: list[str],
    interactions: list[str],
    tmax_mins: Optional[float] = None,
    cmap: str = "tab10",
    figsize: tuple[float, float] = (2.4, 3.0),
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
):
    print("Loading data...")
    data_mutants = []
    data = []
    with h5py.File(data_hdf, "r") as f:
        t_secs = f["t_secs"][...]
        datasets = list(f.keys())
        for dset in datasets:
            if dset == "t_secs":
                continue
            y_t = f[dset][...]
            genotype = f[dset].attrs["genotype"]

            print("Genotype:", genotype)
            acf_min = f[dset].attrs["ACF_min"]
            print(f"ACF min: {acf_min:.3f}")
            frequency = f[dset].attrs["frequency_per_sec"]
            print(f"Period: {1/(60 * frequency):.3f} min")

            data_mutants.append(dset)
            data.append((genotype, y_t))
    t_mins = t_secs / 60.0

    # Reorder into a consistent order
    which_wt = data_mutants.index("WT")
    wt_genotype = data[which_wt][0]
    components = wt_genotype.strip("*").split("::")[0]
    mutants = ["WT"] + list(components)
    mutant_order = np.argsort(data_mutants)[np.argsort(np.argsort(mutants))]
    genotypes, y_ts = zip(*[data[i] for i in mutant_order])

    # Isolate transcription factor dynamics
    print("Isolating transcription factor dynamics...")
    m = len(components)
    mm1 = m - 1
    prots_ts = np.zeros((len(mutants), m, len(t_mins)))
    indices = list(range(m))
    for i, y_t in enumerate(y_ts):
        if i == 0:
            prots_ts[i] = y_t[:, m : m * 2].T
        else:
            missing = i - 1
            mut_indices = np.array(indices[:missing] + indices[missing + 1 :])
            prots_ts[i, mut_indices] = y_t[:, mm1 : mm1 * 2].T

    if tmax_mins is not None:
        tmax_idx = np.searchsorted(t_mins, tmax_mins, side="right")
        t_mins = t_mins[:tmax_idx]
        prots_ts = prots_ts[:, :, :tmax_idx]

    # Plot oscillations
    print("Plotting oscillations...")
    _cmap = colormaps.get_cmap(cmap)
    grammar = SimpleNetworkGrammar(components=components, interactions=interactions)
    for i, (mutant, genotype, prots_t) in enumerate(zip(mutants, genotypes, prots_ts)):
        component_indices = list(range(m))
        if mutant != "WT":
            component_indices.pop(i - 1)
        component_indices = np.array(component_indices)
        # colormap = lambda j: _cmap(component_indices[j])

        fig1, ax1 = plt.subplots(1, 1, figsize=(figsize[0] * 0.6, figsize[1] * 0.6))
        plt.sca(ax1)
        interactions = genotype.split("::")[1]
        plot_network(
            *grammar.parse_genotype(f"*{components}::{interactions}"),
            ax=ax1,
            cmap=cmap,
            # colormap=colormap,
            **_network_kwargs,
        )
        if save:
            today = datetime.today().strftime("%y%m%d")
            fname = f"{today}_mutant{mutant}_network_diagram.{fmt}"
            fpath = Path(save_dir).joinpath(fname).resolve().absolute()
            print("Writing to:", fpath)
            plt.savefig(fpath, dpi=dpi, transparent=True)

        fig2, ax2 = plt.subplots(1, 1, figsize=figsize)

        plt.sca(ax2)
        for j in component_indices:
            ax2.plot(t_mins, prots_t[j], lw=1, color=_cmap(j))

        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("TF Quantity")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
        plt.locator_params(axis="y", tight=True, nbins=3)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax2.set_aspect(aspect=1.0 / ax2.get_data_ratio(), adjustable="box")

        plt.title(f"{mutant} mutant")
        plt.tight_layout()

        if save:
            today = datetime.today().strftime("%y%m%d")
            fname = f"{today}_mutant{mutant}_TF_dynamics.{fmt}"
            fpath = Path(save_dir).joinpath(fname).resolve().absolute()
            print("Writing to:", fpath)
            plt.savefig(fpath, dpi=dpi, transparent=True)

        # Filter the data with a 9-point binomial filter, then compute autocorrelation
        # filtered9 = np.apply_along_axis(binomial9_kernel, -1, prots_t)[:, 4:-4]
        # acorrs_f9 = np.apply_along_axis(autocorrelate, -1, filtered9)
        # t_f9 = t[4:-4]
        # all_minima_idx, all_minima = zip(*[compute_lowest_minima(a) for a in acorrs_f9])
        # which_lowest = np.argmin(all_minima)
        # where_minimum = all_minima_idx[which_lowest]
        # minimum_val = all_minima[which_lowest]
        # corr_time = t_f9[where_minimum]

        # plot_network_quantity_and_acorr(
        #     genotype=genotype,
        #     t=t_f9,
        #     y_t=filtered9,
        #     acorr=acorrs_f9,
        #     corr_time=corr_time,
        #     minimum_val=minimum_val,
        #     plot_dir=save_dir,
        #     figsize=figsize,
        #     save=save,
        #     dpi=dpi,
        #     fmt=fmt,
        #     suffix=f"_{mutant}_mutant",
        #     annotate_acf_min=False,
        #     **_network_kwargs,
        # )

    # assembly_path = [
    #     "*ABCDE::AAa_ACi_BAi_BBa_CBi_CCa",
    #     "*ABCDE::AAa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DDa",
    #     "*ABCDE::AAa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DDa_DEa_EDi",
    # ]
    # assembly_path = [
    #     "*ABCDE::AAa_ABa_BAi_BCi_BDi_BEi_CCa_CDi_DDa_DEi_ECi_EEa",
    # ]
    # for i, genotype in enumerate(assembly_path):
    #     fig, ax = plt.subplots(1, 1, figsize=(figsize[0] * 0.6, figsize[1] * 0.6))
    #     plt.sca(ax)
    #     interactions = genotype.split("::")[1]
    #     plot_network(
    #         *grammar.parse_genotype(f"*{components}::{interactions}"),
    #         ax=ax,
    #         cmap=cmap,
    #         # colormap=colormap,
    #         **_network_kwargs,
    #     )

    #     if save:
    #         today = datetime.today().strftime("%y%m%d")
    #         fname = f"{today}_assembly_path_{i}.{fmt}"
    #         fpath = Path(save_dir).joinpath(fname).resolve().absolute()
    #         print("Writing to:", fpath)
    #         plt.savefig(fpath, dpi=dpi, transparent=True)


if __name__ == "__main__":
    data_hdf = Path(
        # "data/oscillation/231121_5TF_FTO_mutants/231121_mutants_data_5TF_AI+Rep_control.hdf5"  # 2/5 FT, AI+Rep
        # "data/oscillation/231121_5TF_FTO_mutants/231121_mutants_data_5TF_AI+3Rep.hdf5"  # 3/5 FT, AI+3Rep
        "data/oscillation/231121_5TF_FTO_mutants/231121_mutants_data_5TF_3AI+3Rep.hdf5"  # 4/5 FT, 3AI+3Rep
    )

    # save_dir = Path("figures/oscillation/231121_FTO_AI+Rep")
    # save_dir = Path("figures/oscillation/231121_FTO_AI+3Rep")
    save_dir = Path("figures/oscillation/231121_FTO_3AI+3Rep")

    save_dir.mkdir(exist_ok=True)

    main(
        data_hdf=data_hdf,
        components=list("ABCDE"),
        interactions=["activates", "inhibits"],
        tmax_mins=120,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
