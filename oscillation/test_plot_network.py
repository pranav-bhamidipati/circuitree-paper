from circuitree.viz import plot_network
from circuitree.models import SimpleNetworkGrammar
import matplotlib.pyplot as plt
from pathlib import Path


def main(g, save=False, save_dir=None, fmt="png", dpi=300, transparent=False, **kwargs):
    components, activations, inhibitions = SimpleNetworkGrammar.parse_genotype(
        g, nonterminal_ok=True
    )

    fig = plt.figure(figsize=(2, 2))
    plot_network(components, activations, inhibitions, **kwargs)

    if save:
        fname = Path(save_dir).joinpath(f"network.{fmt}").resolve().absolute()
        print(f"Writing to {fname}")
        plt.savefig(fname, dpi=dpi, transparent=transparent)

    ...


if __name__ == "__main__":
    save_dir = Path("figures/oscillation")
    g = "*ABC::AAa_ABa_BAi_BBi_BCi_CAi_CBa_CCa"
    main(
        g,
        save=True,
        save_dir=save_dir,
        cmap="Set2",
        fontsize=14,
        # transparent=True,
        text_kwargs=dict(color="k"),
    )
