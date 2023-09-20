# from concurrent.futures import ThreadPoolExecutor
from circuitree.parallel import search_mcts_in_thread
from datetime import datetime
from functools import partial
from pathlib import Path

import gevent

from oscillation_multithreaded import (
    MultithreadedOscillationTree,
    progress_callback_in_main,
    progress_callback_in_thread,
)

# from time import sleep
#
# def test_threads(mtree, i):
#     mtree.logger.info(f"Thread {i} is running.")
#     sleep(5 - i)
#     mtree.logger.info(f"Thread {i} is done.")


def main(
    save_dir: Path,
    threads: int = 16,
    n_steps_per_thread: int = 10_000,
    max_interactions: int = 12,
    callback_every: int = 1,
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

    # # Uncomment for 3-component oscillation
    # mtree_kw = dict(
    #     root="ABC::",
    #     components=["A", "B", "C"],
    #     interactions=["activates", "inhibits"],
    #     max_interactions=max_interactions,
    #     dt=20.0,  # seconds
    #     nt=2000,
    #     save_dir=save_dir,
    #     threads=threads,
    #     logger=logger,
    # )

    mtree = MultithreadedOscillationTree(**mtree_kw)

    print(f"Making a search tree with {threads} search threads.")
    print(f"Saving results to {save_dir}")
    print()

    # if run_in_main_thread:
    #     start_msg = f"Starting search in main thread."
    #     print(start_msg)
    #     mtree.logger.info(start_msg)
    #     run_search(0)
    # else:
    #     with ThreadPoolExecutor(max_workers=threads) as executor:
    #         start_msg = f"Starting searches in {threads} threads."
    #         print(start_msg)
    #         mtree.logger.info(start_msg)
    #         executor.map(run_search, range(threads))

    now = datetime.now().strftime("%y%m%d_%H%M%S")
    gml_file = save_dir / f"{now}_tree.gml"
    json_file = save_dir / f"{now}_tree.json"

    if run_in_main_thread:
        run_search = partial(
            search_mcts_in_thread,
            mtree=mtree,
            n_steps=n_steps_per_thread,
            callback=progress_callback_in_main,
            callback_every=callback_every,
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
        run_search = partial(
            search_mcts_in_thread,
            mtree=mtree,
            n_steps=n_steps_per_thread,
            callback=progress_callback_in_thread,
            callback_every=callback_every,
            return_metrics=False,
        )
        start_msg = f"Starting searches in {threads} threads...\n"
        print(start_msg)
        mtree.logger.info(start_msg)
        threads = [gevent.spawn(run_search, i) for i in range(threads)]
        gevent.joinall(threads)

        finish_msg = (
            f"Finished searches in {threads} threads. Saving results to {save_dir}"
        )
        print(finish_msg)
        mtree.logger.info(finish_msg)
        mtree.to_file(gml_file, json_file)


if __name__ == "__main__":
    from datetime import date
    import logging

    today = date.today().strftime("%y%m%d")
    save_dir = Path(
        f"~/git/circuitree-paper/data/oscillation/mcts/{today}_5tf_lockfree"
    ).expanduser()
    save_dir.mkdir(exist_ok=True)

    log_file = Path("~/git/circuitree-paper/oscillation/celery/main.log").expanduser()
    logging.basicConfig(
        filename=str(log_file),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info("Running main()")

    main(
        save_dir=save_dir,
        # threads=0,
        # threads=30,
        threads=200,
        n_steps_per_thread=10_000,
        max_interactions=12,
        logger=logger,
        callback_every=10,
    )
