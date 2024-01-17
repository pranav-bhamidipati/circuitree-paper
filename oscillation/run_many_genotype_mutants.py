from pathlib import Path
from typing import Optional
from run_genotype_mutants import main as run_mutants
from multiprocessing.pool import Pool


def main(
    genotypes_csv: Path,
    seed: int = 2023,
    nprocs: int = 1,
    header: bool = True,
    progress: bool = False,
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
    **kwargs,
):
    genotypes = genotypes_csv.read_text().splitlines()
    if header:
        genotypes.pop(0)
    with Pool(nprocs) as pool:
        for i, genotype in enumerate(genotypes):
            run_mutants(
                genotype=genotype,
                seed=seed,
                pool=pool,
                progress=progress,
                save=save,
                save_dir=save_dir,
                suffix=f"{suffix}_{i}",
                **kwargs,
            )


if __name__ == "__main__":
    # save_dir = Path("data/oscillation/231120_5TF_most_robust_mutants")
    # data_csv = save_dir.joinpath("231120_5tf_robust_states.csv")

    data_csv = Path(
        "data/oscillation/231121_5tf_fault_tolerance_states/231121_5tf_FT_states.csv"
    )
    save_dir = data_csv.parent

    main(
        genotypes_csv=data_csv,
        seed=2023,
        # nprocs=1,
        nprocs=6,
        progress=True,
        save=True,
        save_dir=save_dir,
        # suffix=suffix,
    )
