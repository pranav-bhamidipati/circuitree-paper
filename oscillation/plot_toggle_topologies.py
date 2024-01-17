from pathlib import Path
from plot_ai_suppressor_comparison import plot_matching_samples_from_ttable_data


def main(
    tranposition_table_parquet: Path,
    data_dir: Path,
    genotypes: list[str] = [
        "*ABC::ABi_BAi",
        "*ABC::ABa_ACi_CAi_CBi",
    ],
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

    save_dir = Path("figures/oscillation/toggle_comparison")
    save_dir.mkdir(exist_ok=True)

    main(
        tranposition_table_parquet=ttable_parquet,
        data_dir=data_dir,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
