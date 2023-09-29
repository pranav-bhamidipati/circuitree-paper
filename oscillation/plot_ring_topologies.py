from datetime import date
from typing import Optional
from circuitree.viz import plot_network
import matplotlib.pyplot as plt
from pathlib import Path

from oscillation import OscillationGrammar

_network_kwargs = dict(
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
)


def main(
    figsize: tuple = (2.0, 5.0),
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "svg",
    dpi: int = 300,
    **kwargs,
):
    # Plotting parameters
    network_kwargs = _network_kwargs | kwargs
    grammar = OscillationGrammar(
        components=["A", "B", "C"],
        interactions=["activates", "inhibits"],
    )

    ring_topologies = ["AB::ABa_BAi", "ABC::ABi_BCi_CAi", "ABC::ABa_BCa_CAi"]

    gridspec_kw = dict(height_ratios=[1, 1.5, 1.5])
    # fig = plt.figure(figsize=figsize)
    # for i, ring in enumerate(ring_topologies):
    # ax = fig.add_subplot(3, 1, i + 1, gridspec_kw=gridspec_kw)
    fig, axs = plt.subplots(3, 1, figsize=figsize, **gridspec_kw)
    for ax, ring in zip(axs, ring_topologies):
        plt.sca(ax)

        plot_network(
            *grammar.parse_genotype(ring, nonterminal_ok=True), ax=ax, **network_kwargs
        )

    plt.tight_layout()

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_ring_topologies.{fmt}"
        fpath = Path(save_dir).joinpath(fname).resolve().absolute()
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    save_dir = Path("figures/oscillation")
    main(
        save=True,
        save_dir=save_dir,
        fmt="eps",
        plot_labels=False,
    )
