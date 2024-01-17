from pathlib import Path
from plot_oscillation_trace_and_acf import plot_matching_samples_from_hdf5


def main(
    data_hdf: Path,
    genotypes: list[str],
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    figsize: tuple[float, float] = (3.0, 3.5),
):
    plot_matching_samples_from_hdf5(
        genotypes=genotypes,
        hdf=data_hdf,
        save=save,
        save_dir=save_dir,
        fmt=fmt,
        dpi=dpi,
        figsize=figsize,
    )


if __name__ == "__main__":
    # data_hdf = Path(
    #     "data/oscillation/231206_AI_variations"
    #     "/231206_simulation_data_3TF_AI_variations.hdf5"
    # )
    data_hdf = Path(
        "data/oscillation/231207_AI_variations"
        "/231207_simulation_data_3TF_AI_variations.hdf5"
    )
    genotypes = [
        "*ABC::ABa_BAi",
        "*ABC::AAa_ABa_BAi",
        "*ABC::ABa_BAi_CBi",
        "*ABC::ABa_BAi_CAa",
        "*ABC::AAa_ABa_BAi_CBi",
    ]

    save_dir = Path("figures/oscillation/ai_comparisons")
    save_dir.mkdir(exist_ok=True)

    main(
        data_hdf=data_hdf,
        genotypes=genotypes,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
