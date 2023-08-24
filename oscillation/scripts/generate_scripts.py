from math import ceil
from pathlib import Path
from tqdm import trange

_template_default = Path(
    "/home/pbhamidi/git/circuitree/models/oscillation/scripts/submit_bfs_arrayjob_template.sh"
)
_output_default = Path(
    "/home/pbhamidi/git/circuitree/models/oscillation/scripts/arrayjobs"
)


def main(
    n_topologies: int,
    n_tasks_per_topology: int,
    arraysize: int = 1000,
    skip: int = 2,
    jobname: str = "CircuiTree_Oscillation_ArrayJob",
    tasklimit: int = 2000,
    template_path: Path = _template_default,
    output_dir: Path = _output_default,
    prompt: bool = True,
):
    """Generates array job scripts for the BFS search."""

    template = Path(template_path).read_text()
    n_total_tasks = n_topologies * n_tasks_per_topology
    n_submissions = ceil(n_total_tasks / arraysize)

    if prompt:
        make_scripts = input(
            f"Are you sure you want to generate {n_submissions} scripts? [y/N]: "
        )
        if make_scripts not in ["y", "Y"]:
            print("Aborting.")
            return

    n_digits = len(str(n_submissions - 1))
    for i in trange(n_submissions, desc="Generating scripts"):
        this_idx_as_str = str(i).zfill(n_digits)
        this_script = (
            output_dir.joinpath(f"submit_bfs_arrayjob_{this_idx_as_str}.sh")
            .resolve()
            .absolute()
        )
        if i >= (n_submissions - skip):
            next_script = ""
        else:
            next_idx_as_str = str(i + skip).zfill(n_digits)
            next_script = (
                output_dir.joinpath(f"submit_bfs_arrayjob_{next_idx_as_str}.sh")
                .resolve()
                .absolute()
            )

        first_task_index = i * arraysize
        array_limit = min(arraysize, n_total_tasks - first_task_index) - 1

        out = template.replace("__jobname__", jobname)
        out = out.replace("__jobnum__", str(i))
        out = out.replace("__arraylimit__", str(array_limit))
        out = out.replace("__tasklimit__", str(tasklimit))
        out = out.replace("__index__", str(i * arraysize))
        out = out.replace("__next_jobnum__", str(i + skip))
        out = out.replace("__next_script__", str(next_script))
        this_script.write_text(out)


if __name__ == "__main__":
    
    # For runs used to gather data on computational complexity
    #   - run time and number of Gillespie iterations taken
    niter_template = _template_default.parent.joinpath("submit_niter_arrayjob_template.sh")
    niter_output_dir = _output_default.parent.joinpath("arrayjobs_niter")
    
    main(
        # ### Default args
        # n_topologies=3411,
        # n_tasks_per_topology=100,
        # skip=2,
        # arraysize=1000,
        # tasklimit=2000,
        # ### Args below are for testing
        # n_topologies=10,
        # n_tasks_per_topology=3,
        # arraysize=5,
        # tasklimit=10,
        # skip=2,
        # prompt=False,
        ### Args below are for computational complexity
        jobname="CircuiTree_Complexity_ArrayJob",
        n_topologies=3411,
        n_tasks_per_topology=1000,
        skip=2,
        arraysize=1000,
        tasklimit=2000,
    )
