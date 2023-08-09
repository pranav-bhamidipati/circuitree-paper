import numpy as np
from oscillation.oscillation import TFNetworkModel
from time import perf_counter
import timeit

code = "*ABC::AiB_BiC_CiA"
dt = 20.0
# nt = 2000
nt = 100
time_points = np.linspace(0, dt * nt, nt + 1)

model = TFNetworkModel(code, seed=0)
model.initialize_ssa(dt, nt)

...

# pop0, _ = model.ssa.draw_random_initial_and_params()
# params = model.ssa.DEFAULT_PARAMS
# ssa_params = model.ssa.package_params_for_ssa(params)

# print("Starting a vanilla Gillespie simulation...")
# y_t = model.ssa.gillespie_trajectory(pop0, *ssa_params)
# acorr2pk = model.get_secondary_autocorrelation_peaks(time_points, y_t)
# print(f"\tAutocorrelation peak: {acorr2pk}")

# print("Starting a tau-leaping simulation...")
# y_t = model.ssa.gillespie_tau_leaping(pop0, *ssa_params)
# acorr2pk = model.get_secondary_autocorrelation_peaks(time_points, y_t)
# print(f"\tAutocorrelation peak: {acorr2pk}")

...


# # Testing the computational cost of each tau-leap step
# from gillespie import take_tau_leap, take_gillespie_step

# Uz1 = model.ssa.U
# U = Uz1[:-1]
# Usq = U**2
# m4a3r3, m2ar = U.shape
# events = np.zeros((m4a3r3, 1)).astype(np.int64)
# tau_mtx = np.zeros((2, m2ar)).astype(np.float64)
# propensities = np.zeros(m4a3r3).astype(np.float64)
# event = m4a3r3
# epsilon = 0.03
# highest_order_rxn_idx = model.ssa.highest_order_rxn_idx
# tau_mtx = np.zeros((2, m2ar)).astype(np.float64)
# activations_left = model.ssa.activations_left
# activations_right = model.ssa.activations_right
# inhibitions_left = model.ssa.inhibitions_left
# inhibitions_right = model.ssa.inhibitions_right
# m = model.ssa.m
# a = model.ssa.a
# r = model.ssa.r
# step = lambda: take_gillespie_step(
#     time_points[0],
#     pop0,
#     event,
#     propensities,
#     Uz1,
#     activations_left,
#     activations_right,
#     inhibitions_left,
#     inhibitions_right,
#     m,
#     a,
#     r,
#     *ssa_params,
# )
# leap = lambda: take_tau_leap(
#     pop0,
#     events,
#     propensities,
#     U,
#     Usq,
#     epsilon,
#     highest_order_rxn_idx,
#     tau_mtx,
#     activations_left,
#     activations_right,
#     inhibitions_left,
#     inhibitions_right,
#     m,
#     a,
#     r,
#     *ssa_params,
# )

# print("Timing a Gillespie step...")
# step(); # compile
# step_times = timeit.repeat(step, number=10000, repeat=7)
# print(f"\tTime: {np.mean(step_times):.5f} +/- {np.std(step_times):.5f} s")

# print("Timing a tau leap...")
# leap(); # compile
# leap_times = timeit.repeat(leap, number=10000, repeat=7)
# print(f"\tTime: {np.mean(leap_times):.5f} +/- {np.std(leap_times):.5f} s")

...

# m = model.n_components
# a = model.n_activations
# r = model.n_inhibitions

# pop0 = np.zeros(model.n_species).astype(np.int64)
# pop0[m : 2 * m] = 10
# params = model.ssa.DEFAULT_PARAMS


# activations_left, activations_right = model.activations.reshape((a, 2)).T
# inhibitions_left, inhibitions_right = model.inhibitions.reshape((r, 2)).T
# ssa_params = model.ssa.package_params_for_ssa(params)

# prop_args = (
#     pop0,
#     m,
#     a,
#     # r,
#     activations_left,
#     activations_right,
#     inhibitions_left,
#     inhibitions_right,
#     *ssa_params,
# )

# model.ssa.get_propensities(*prop_args)

...

# pop0 = np.zeros(model.n_species).astype(np.int64)
# params = model.ssa.DEFAULT_PARAMS  # repressilator parameters

# y_t = model.run_ssa_with_params(pop0, params)

...

start1 = perf_counter()
_pop0, _params, y_t1 = model.run_ssa_random_params()
stop1 = perf_counter()

print(f"Time for 1st run: {stop1 - start1:.5f} s")

# start2 = perf_counter()
# y_t2 = model.run_ssa_with_params(_pop0, _params)
# stop2 = perf_counter()

# print(f"Time for 2nd run: {stop2 - start2:.5f} s")

start2 = perf_counter()
y_t2 = model.run_ssa_with_params(_pop0, _params, tau_leap=True)
stop2 = perf_counter()

print(f"Time for 1st run with tau leaping: {stop2 - start2:.5f} s")

...

#### Specific param sets with problems


# weird_params = (4.182487464232694, 197.0436732236758, 36.281113993516804, 0.5977693255166171, 1.0, 5.334891492520546e-05, 0.5977693255166171, 0.21877604444356308, 0.004632512227321745, 0.011246850498897381)

# weird_p0 = [ 0  0  0  7 16  9  0  0  0]
# weird_params = np.array((36.39506607927666, 246.6137601961709, 201.997632857264, 0.563536021488523, 1.0, 5.002625765416854e-05, 0.563536021488523, 0.20402913038870174, 0.0015213435178751023, 0.01876917414067622))
weird_p0 = np.array([0, 0, 0, 12, 10, 6, 0, 0, 0])
weird_params = np.array(
    (
        0.10302688405217551,
        2.7094158225836207,
        0.34734675643704116,
        0.753038245346722,
        1.0,
        0.0011056941021836038,
        0.753038245346722,
        0.18909289328640683,
        0.016819497386391317,
        0.0011576611859387442,
    )
)

start2 = perf_counter()
y_t_weird = model.run_ssa_with_params(weird_p0, weird_params, tau_leap=True)
stop2 = perf_counter()
print(f"Weird run finished in time: {stop2 - start2:.5f} s")

...

inits_and_param_sets = [model.ssa.draw_random_initial_and_params() for _ in range(100)]
times = []
for i, (pop0, params) in enumerate(inits_and_param_sets):
    start = perf_counter()
    model.run_ssa_with_params(pop0, params, tau_leap=False)
    stop = perf_counter()
    times.append(stop - start)

    print(f"Max so far: {max(times)}")

...

inits_and_param_sets = [model.ssa.draw_random_initial_and_params() for _ in range(10)]

for i, (pop0, params) in enumerate(inits_and_param_sets):
    print(f"Run #{i + 1}")
    leap_repeats = timeit.repeat(
        lambda: model.run_ssa_with_params(pop0, params, tau_leap=True),
        number=1,
        repeat=5,
    )
    print(
        f"\tTau-leaping (mean +/- sd): {np.mean(leap_repeats):.5f} +/- {np.std(leap_repeats):.5f} s"
    )
    step_repeats = timeit.repeat(
        lambda: model.run_ssa_with_params(pop0, params), number=1, repeat=5
    )
    print(
        f"\tVanilla (mean +/- sd): {np.mean(step_repeats):.5f} +/- {np.std(step_repeats):.5f} s"
    )

...

pop0s, param_sets, y_ts = model.run_batch(10)

...

# n = 10

# pop0s = np.zeros((n, ssa.n_species)).astype(np.int64)
# param_sets = np.zeros((n, ssa.n_params)).astype(np.float64)
# y_ts = np.zeros((n, ssa.nt, ssa.n_species)).astype(np.int64)
# for i in range(n):
#     pop0, params, y_t = ssa.run_random_sample()
#     pop0s[i] = pop0
#     param_sets[i] = params
#     y_ts[i] = y_t


# ...
