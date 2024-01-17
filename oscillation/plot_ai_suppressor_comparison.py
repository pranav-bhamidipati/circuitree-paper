from pathlib import Path
from plot_oscillation_trace_and_acf import plot_matching_samples_from_ttable_data


def main(
    tranposition_table_parquet: Path,
    data_dir: Path,
    genotypes: list[str] = ["*ABC::AAa_ABa_BAi", "*ABC::AAa_ABa_BAi_CBi"],
    **kwargs,
):
    plot_matching_samples_from_ttable_data(
        transposition_table_parquet=tranposition_table_parquet,
        data_dir=data_dir,
        genotypes=genotypes,
        **kwargs,
    )


if __name__ == "__main__":
    ttable_parquet = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    data_dir = Path("data/oscillation/bfs_230710_hpc/extras")

    save_dir = Path("figures/oscillation/ai_comparisons")
    save_dir.mkdir(exist_ok=True)

    # genotypes = [
    #     # "*ABC::ABa_BAi",
    #     "*ABC::AAa_ABa_BAi",
    #     "*ABC::ABa_BAi_CBi",
    #     # "*ABC::ABa_BAi_CAa",
    #     "*ABC::AAa_ABa_BAi_CBi",
    # ]

    main(
        tranposition_table_parquet=ttable_parquet,
        # genotypes=genotypes,
        data_dir=data_dir,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
