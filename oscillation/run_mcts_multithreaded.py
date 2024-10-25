from typing import Optional
from datetime import datetime
from functools import partial
from pathlib import Path

from oscillation_multithreaded import (
    MultithreadedOscillationTree,
    progress_and_backup_in_thread,
)


def main(
    save_dir: Path,
    param_sets_csv: Path,
    threads: int,
    n_components: int,
    n_steps: int = 100_000,
    n_exhausted: Optional[int] = None,
    exploration_constant: Optional[float] = None,
    max_interactions: int = 12,
    callback_every: int = 1,
    progress_every: int = 10,
    backup_prefix: str = "",
    backup_every: int = 10800,
    n_tree_backups: int = 1000,
    cache_maxsize: int = 2**20,
    logger=None,
    dry_run: bool = False,
    run_as_main: bool = False,
    seed: Optional[int] = None,
    dt=20.0,  # seconds
    nt=2160,
    nchunks: int = 5,
    success_threshold: float = 0.01,
    autocorr_threshold: float = -0.4,
    **kwargs,
):
    print(f"Making a search tree with {threads} search threads.")
    print(f"Saving results to {save_dir}")
    print()

    components = "ABCDEFGHIJK"[:n_components]
    tree = MultithreadedOscillationTree(
        root=f"{components}::",
        components=list(components),
        interactions=["activates", "inhibits"],
        seed=seed,
        n_exhausted=n_exhausted,
        max_interactions=max_interactions,
        exploration_constant=exploration_constant,
        threads=threads,
        save_dir=save_dir,
        param_sets_csv=param_sets_csv,
        logger=logger,
        dt=dt,
        nt=nt,
        nchunks=nchunks,
        success_threshold=success_threshold,
        autocorr_threshold=autocorr_threshold,
        cache_maxsize=cache_maxsize,
        tz_offset=-7,
    )

    if run_as_main:
        threads = 0
        callback = partial(
            progress_and_backup_in_thread,
            progress_every=1,
            backup_every=1,
            n_tree_backups=n_tree_backups,
            db_backup_prefix=backup_prefix,
            dry_run=dry_run,
            dump_results=True,
        )

        start_msg = "Running search in main thread...\n"
        print(start_msg)
        tree.logger.info(start_msg)
        tree.search_mcts(
            n_steps,
            callback=callback,
            callback_every=1,
            callback_before_start=True,
        )
    else:
        callback = partial(
            progress_and_backup_in_thread,
            progress_every=progress_every,
            backup_every=backup_every,
            n_tree_backups=n_tree_backups,
            db_backup_prefix=backup_prefix,
            dry_run=dry_run,
            dump_results=True,
        )

        start_msg = f"Starting MCTS using {threads} threads...\n"
        print(start_msg)
        tree.logger.info(start_msg)
        tree.search_mcts_parallel(
            n_steps,
            threads,
            callback=callback,
            callback_every=callback_every,
            callback_before_start=True,
        )

        if dry_run:
            finish_msg = (
                f"Finished a dry run using {threads} threads. Results not saved."
            )
        else:
            finish_msg = (
                f"Finished MCTS using {threads} threads. Results saved to: {save_dir}"
            )
        print(finish_msg)
        tree.logger.info(finish_msg)


if __name__ == "__main__":
    from datetime import datetime
    import logging

    # Default is sqrt(2)
    # exploration_constant = 2.0
    max_interactions = 15
    # run_prefix = f"5tf_exhaustion_mutationrate0.5_maxinteractions15_"
    run_prefix = f"3tf_params10k_exhaustion100_"

    param_sets_csv = Path(
        "~/git/circuitree-paper/data/oscillation_asymmetric_params/"
        # "241023_param_sets_10000_5tf.csv"
        # "241023_param_sets_10000_3tf.csv"
        "241025_param_sets_10000_5tf_pmut0.5.csv"
    ).expanduser()

    now = datetime.now().strftime("%y%m%d-%H-%M-%S")
    save_dir = (
        f"~/git/circuitree-paper/data/oscillation_asymmetric_params/mcts/"
        f"{now}_{run_prefix}"
        # f"_exploration{exploration_constant:.3f}"
    ).strip("_")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir()

    log_file = Path(
        "~/git/circuitree-paper/logs/oscillation_asymmetric_params/mcts/"
        "worker-logs/main.log"
    ).expanduser()
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(log_file),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        # level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)
    logger.info("Running main() program.")

    main(
        save_dir=save_dir,
        param_sets_csv=param_sets_csv,
        # threads=0,
        threads=600,
        n_components=5,
        n_steps=10_000_000,
        n_exhausted=10_000,
        # exploration_constant=exploration_constant,
        max_interactions=max_interactions,
        # callback_every=1,
        callback_every=10,
        # progress_every=1,
        progress_every=10,
        backup_prefix=run_prefix,
        # backup_every=300,
        # backup_every=3600,
        backup_every=7200,
        # dry_run=True,
        # run_as_main=True,
        seed=2024,
        logger=logger,
    )
