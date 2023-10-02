from typing import Optional
from circuitree.parallel import search_mcts_in_thread
from datetime import datetime
from functools import partial
from pathlib import Path

import gevent

from oscillation_multithreaded import (
    MultithreadedOscillationTree,
    progress_and_backup_in_thread2,
)


def main(
    save_dir: Path,
    backup_dir: Optional[Path] = None,
    threads: int = 25,
    n_steps_per_thread: int = 1_000,
    max_interactions: int = 4,
    callback_every: int = 1,
    backup_every: int = 25,
    n_tree_backups: int = 2,
    now: str = None,
    logger=None,
):
    if threads == 0:
        run_in_main_thread = True
        threads = 1
    else:
        run_in_main_thread = False

    mtree_kw = dict(
        root="ABCDE::",
        components=["A", "B", "C", "D", "E"],
        interactions=["activates", "inhibits"],
        max_interactions=max_interactions,
        dt=20.0,  # seconds
        nt=2000,
        save_dir=save_dir,
        threads=threads,
        logger=logger,
    )

    mtree = MultithreadedOscillationTree(**mtree_kw)

    print(f"Making a search tree with {threads} search threads.")
    print(f"Saving results to {save_dir}")
    print()

    now = now or datetime.now().strftime("%y%m%d_%H%M%S")
    # gml_file = save_dir / f"{now}_tree.gml"
    # json_file = save_dir / f"{now}_tree.json"

    if run_in_main_thread:
        callback = partial(
            progress_and_backup_in_thread2,
            backup_dir=backup_dir,
            backup_every=1,
            # gml_file=gml_file,
            # json_file=json_file,
            n_tree_backups_to_keep=n_tree_backups,
            # keep_single_gml_backup=True,
            backup_results=True,
        )
        run_search = partial(
            search_mcts_in_thread,
            mtree=mtree,
            n_steps=n_steps_per_thread,
            callback=callback,
            callback_every=1,
            return_metrics=False,
        )
        start_msg = "Running search in main thread...\n"
        print(start_msg)
        mtree.logger.info(start_msg)
        run_search(0)

        finish_msg = (
            f"Finished searches in {threads} threads. Saving results to {save_dir}"
        )
        print(finish_msg)
        mtree.logger.info(finish_msg)

    else:
        gml_file = backup_dir / f"{now}_tree_backup.gml"
        json_file = backup_dir / f"{now}_tree_metadata.json"
        callback = partial(
            progress_and_backup_in_thread2,
            backup_dir=backup_dir,
            backup_every=backup_every,
            gml_file=gml_file,
            json_file=json_file,
            keep_single_gml_backup=True,
            backup_results=True,
        )

        run_search = partial(
            search_mcts_in_thread,
            mtree=mtree,
            n_steps=n_steps_per_thread,
            callback=callback,
            callback_every=callback_every,
            return_metrics=False,
        )
        start_msg = f"Starting searches in {threads} threads...\n"
        print(start_msg)
        mtree.logger.info(start_msg)
        gthreads = [gevent.spawn(run_search, i) for i in range(threads)]
        gevent.joinall(gthreads)

        finish_msg = (
            f"Finished searches in {threads} threads. Saving results to {save_dir}"
        )
        print(finish_msg)
        mtree.logger.info(finish_msg)
        mtree.to_file(gml_file, json_file)


if __name__ == "__main__":
    from datetime import datetime
    import logging

    now = datetime.now().strftime("%y%m%d-%H-%M-%S")
    save_dir = Path(
        f"~/git/circuitree-paper/data/oscillation/mcts/{now}_5tf_lockfree_test_backups"
    ).expanduser()
    save_dir.mkdir()
    backup_dir = save_dir.joinpath("backups")
    backup_dir.mkdir()

    log_file = Path(f"~/git/circuitree-paper/logs/worker-logs/test.log").expanduser()
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
        backup_dir=backup_dir,
        threads=0,
        backup_every=1,
        n_tree_backups=4,
        # threads=25,
        n_steps_per_thread=50,
        max_interactions=4,
        logger=logger,
        callback_every=1,
        now=now,
    )
