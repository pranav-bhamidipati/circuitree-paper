from functools import partial
from time import perf_counter
import numpy as np
import multiprocessing as mp
from psutil import cpu_count
from gillespie import (
    GillespieSSA,
    make_matrices_for_ssa,
    SAMPLING_RANGES,
    DEFAULT_PARAMS,
)


def run_with_tau_cost(tau_cost, ssa: GillespieSSA, pop0):
    ssa.tau_leap_cost = tau_cost
    start = perf_counter()
    ssa.run_with_params_tau_leap(pop0, ssa.DEFAULT_PARAMS)
    end = perf_counter()
    return end - start


def run_gillespie(_, ssa: GillespieSSA, pop0):
    start = perf_counter()
    ssa.run_with_params(pop0, ssa.DEFAULT_PARAMS)
    end = perf_counter()
    return end - start


def main(tau_costs, replicates=1, n_threads=None):
    # Activator-inhibitor circuit
    # 2 components A and B. AAa_ABa_BAi
    nc = 2
    Am, Rm, U = make_matrices_for_ssa(
        nc, activations=[[0, 0], [0, 1]], inhibitions=[[1, 0]]
    )

    ssa = GillespieSSA(
        seed=0,
        n_species=2,
        activation_mtx=Am,
        inhibition_mtx=Rm,
        update_mtx=U,
        dt=20.0,
        nt=100,
        mean_mRNA_init=10.0,
        PARAM_RANGES=SAMPLING_RANGES,
        DEFAULT_PARAMS=DEFAULT_PARAMS,
        tau_leap_cost=0.0,
    )

    pop0 = np.zeros(ssa.n_species, dtype=np.int64)
    pop0[nc : nc * 2] = 10

    ...

    ### Takes 185515 Gillespie steps
    # _ = ssa.run_with_params(pop0, ssa.DEFAULT_PARAMS)

    ### Takes 77644 tau-leaps
    # _ = ssa.run_with_params_tau_leap(pop0, ssa.DEFAULT_PARAMS)

    if n_threads is None:
        n_threads = cpu_count(logical=True)
    else:
        n_threads = min(n_threads, cpu_count(logical=True))
    print(f"Assembling pool of {n_threads} processes")

    run_one_tau = partial(run_with_tau_cost, ssa=ssa, pop0=pop0)
    run_one_gill = partial(run_gillespie, ssa=ssa, pop0=pop0)

    inputs = np.tile(tau_costs, replicates)
    with mp.Pool(n_threads) as pool:
        print("Running once w/ tau-leaping to compile")
        _tau_compile = pool.map(run_one_tau, tau_costs)

        print("Running with various tau-leaping settings")
        results = pool.map(run_one_tau, inputs)

        print("Comparing to vanilla Gillespie. Running once (per thread) to compile")
        _gill_compile = pool.map(run_one_gill, range(n_threads))
        gill_results = pool.map(run_one_gill, range(replicates))

    times = np.array(results).reshape(replicates, -1).T
    gill_times = np.array(gill_results)
    print(
        f"Tau-leaping runtimes:",
        *[
            f"Cost={c:.3f} --> {np.mean(ts):.4f} +/- {np.std(ts):.4f}"
            for c, ts in zip(tau_costs, times)
        ],
        sep="\n\t",
    )
    print("Vanilla Gillespie runtime:")
    print(f"\t{np.mean(gill_times):.4f} +/- {np.std(gill_times):.4f}")
    ...


if __name__ == "__main__":
    ...

    tau_costs = np.zeros(16)
    tau_costs[1:15] = np.linspace(1, 2, 14)
    tau_costs[15] = 1000
    main(tau_costs=tau_costs, replicates=100, n_threads=None)

    ...
