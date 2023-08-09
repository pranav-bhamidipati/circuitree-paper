from datetime import date
from functools import reduce
from typing import Iterable, Mapping, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib_venn as venn


def main(
    data_csv: Path,
    motifs: Iterable[str],
    osc_threshold: float = 0.01,
    bg_cmap: str = "Reds",
    vmin: float = 0.0,
    vmax: float = 1.0,
    save_dir: Optional[Path] = None,
    figsize: tuple = (3.6, 3.0),
    save: bool = False,
    fmt: str = "png",
    dpi: int = 300,
    state_col: str = "state",
    plot_colorbar: bool = True,
):
    if len(motifs) not in (2, 3):
        raise ValueError("Must provide 2 or 3 motifs")

    df = pd.read_csv(data_csv, index_col=0)
    df.sort_values("p_oscillation", inplace=True, ascending=False)
    df_osc = df.loc[df["p_oscillation"] >= osc_threshold]

    if state_col not in df.columns:
        raise ValueError(f"State column '{state_col}' not in dataframe columns")
    if any(m not in df.columns for m in motifs):
        raise ValueError("Motifs must be in dataframe columns")

    # Get sets of states that have each motif
    states_with_motif = [set(df_osc[state_col][df_osc[m]]) for m in motifs]

    # Order of combinations for venn diagram is (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    combo_codes = ["100", "010", "110", "001", "101", "011", "111"]
    which_motif_combos = [(0,), (1,), (0, 1), (2,), (0, 2), (1, 2), (0, 1, 2)]
    pct_osc = []
    for combo in which_motif_combos:
        all_mask = df[motifs[combo[0]]]
        osc_mask = df_osc[motifs[combo[0]]]
        for which_m in combo[1:]:
            m = motifs[which_m]
            all_mask = all_mask & df[m]
            osc_mask = osc_mask & df_osc[m]
        for i in range(3):
            if i not in combo:
                all_mask = all_mask & ~df[motifs[i]]
                osc_mask = osc_mask & ~df_osc[motifs[i]]
        n_states = all_mask.sum()
        n_osc = osc_mask.sum()
        pct_osc.append(n_osc / max(n_states, 1))

    # pct_osc = [
    #     n_osc / max(n_tot, 1) for n_osc, n_tot in zip(n_osc_per_comb, n_states_per_comb)
    # ]

    cmap = plt.get_cmap(bg_cmap)
    vrange = vmax - vmin
    bg_colors = [cmap(min(max(vmin, p - vmin) / vrange, vmax)) for p in pct_osc]

    fig, ax = plt.subplots(figsize=figsize)
    vd = venn.venn3_unweighted(states_with_motif, set_labels=motifs, ax=ax)
    for code, color in zip(combo_codes, bg_colors):
        vd.get_patch_by_id(code).set_color(color)
        # vd.get_patch_by_id(code).set_edgecolor("none")
        vd.get_patch_by_id(code).set_alpha(1.0)

    if plot_colorbar:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap),
            ax=ax,
            orientation="vertical",
            fraction=0.1,
            pad=0.1,
            shrink=0.4,
        )
        cbar.ax.set_title("% of total", size=8, y=1.05)
        cbar.ax.tick_params(labelsize=8)

        xlim = ax.get_xlim()
        xlim = xlim[0], xlim[1] - 0.05 * (xlim[1] - xlim[0])
        ax.set_xlim(xlim)

    # "Oscillators" with none of the motifs
    any_motifs = reduce(lambda x, y: x | y, (df_osc[m] for m in motifs))
    n_osc_no_motifs = (~any_motifs).sum()
    plt.text(
        x=0.1,
        y=0.1,
        s=f"No motifs\n{n_osc_no_motifs}",
        ha="center",
        va="bottom",
        size=8,
        transform=ax.transAxes,
    )

    # Total size of state space
    n_states = df.shape[0]
    plt.text(
        x=0.82,
        y=0.1,
        s=f"Total #\ntopologies = {n_states}",
        ha="left",
        va="bottom",
        size=8,
        transform=ax.transAxes,
    )
    plt.text(
        x=0.8,
        y=-0.05,
        s="# oscillators\n" + r"($Q>0.01$) = " + f"{df_osc.shape[0]}",
        ha="left",
        va="bottom",
        size=8,
        transform=ax.transAxes,
    )

    # Total number of oscillators and total size of state space
    # n_oscillators = df_osc.shape[0]
    plt.suptitle(f"Motif representation among oscillators", size=10)

    plt.tight_layout()

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_mcts_oscillation_motifs_venn_diagram.{fmt}"
        target = save_dir.joinpath(fname).resolve().absolute()
        print(f"Writing to: {target}")
        plt.savefig(target, dpi=dpi)

    ...


if __name__ == "__main__":
    data_csv = Path("data/oscillation/230717_motifs.csv")
    save_dir = Path("figures/oscillation/")
    save_dir.mkdir(exist_ok=True)
    motifs = ("AI", "AAI", "III")

    main(
        data_csv=data_csv,
        save_dir=save_dir,
        motifs=motifs,
        save=True,
        vmax=0.4,
    )
