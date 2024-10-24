from itertools import cycle, islice, repeat
from typing import Iterable, Optional
from numba import njit
import numpy as np
from scipy.stats import truncnorm

"""
    This module contains numerical code (largely JIT-compiled) for running a 
    stochastic simulation algorithm using Gillespie sampling.
    
    This was adapted from the code in the `biocircuits` package written by Justin Bois.
    
"""

__all__ = [
    "make_matrices_for_ssa",
    "package_params_for_ssa",
    "DEFAULT_PARAMS",
    "SAMPLING_RANGES",
    "GillespieSSA",
]

DEFAULT_TIME_STEP_SECONDS = 20.0
DEFAULT_N_TIME_POINTS = 2000

DEFAULT_PARAMS = np.array(
    [
        1.0,  # k_on
        99.0,  # k_off_1
        9.9,  # k_off_2
        0.05,  # km_unbound
        8.0,  # km_act
        5e-4,  # km_rep
        0.05,  # km_act_rep
        0.167,  # kp
        0.025,  # gamma_m
        0.025,  # gamma_p
    ],
    dtype=np.float64,
)


PARAM_NAMES = [
    "k_on",
    "k_off_1",
    "k_off_2",
    "km_unbound",
    "km_act",
    "km_rep",
    "km_act_rep",
    "kp",
    "gamma_m",
    "gamma_p",
]


DISSOCIATION_RATE_SUM = 100.0

SAMPLING_RANGES = np.array(
    [
        (-2.0, 4.0),  # log10_kd_1
        (0.0, 0.25),  # kd_2_1_ratio
        (0.0, 1.0),  # km_unbound
        (1.0, 10.0),  # km_act
        (0.0, 5.0),  # nlog10_km_rep_unbound_ratio
        (0.015, 0.25),  # kp
        (0.001, 0.1),  # gamma_m
        (0.001, 0.1),  # gamma_p
    ],
    dtype=np.float64,
)

SAMPLED_VAR_NAMES = [
    "log10_kd_1",
    "kd_2_1_ratio",
    "km_unbound",
    "km_act",
    "nlog10_km_rep_unbound_ratio",
    "kp",
    "gamma_m",
    "gamma_p",
]

SAMPLED_VAR_MATHTEXT = [
    r"$\kappa_1$",
    r"$\kappa_2$",
    r"$k'_{m,\mathrm{unbound}}$",
    r"$k'_{m,\mathrm{act}}$",
    r"$r_\mathrm{rep}$",
    r"$k'_p$",
    r"$\gamma'_m$",
    r"$\gamma'_p$",
]

MEAN_INITIAL_POPULATION = 10.0


@njit
def convert_sampled_quantities_to_params(sampled, param_ranges):
    """Convert the sampled quantities (intrinsic properties) of the system to parameter
    values used in simulation. Some parameters are derived from sampled quantities,
    others are sampled directly."""
    (
        log10_kd_1,
        kd_2_1_ratio,
        km_unbound,
        km_act,
        nlog10_km_rep_unbound_ratio,
        kp,
        gamma_m,
        gamma_p,
    ) = sampled

    # Calculate derived parameters
    kd_1 = 10**log10_kd_1
    k_off_1 = DISSOCIATION_RATE_SUM * kd_1 / (1 + kd_1)
    k_on = DISSOCIATION_RATE_SUM - k_off_1
    k_off_2 = k_off_1 * kd_2_1_ratio

    km_rep = km_unbound * 10**-nlog10_km_rep_unbound_ratio

    # activation and repression together have no effect on transcription
    km_act_rep = km_unbound

    params = (
        k_on,
        k_off_1,
        k_off_2,
        km_unbound,
        km_act,
        km_rep,
        km_act_rep,
        kp,
        gamma_m,
        gamma_p,
    )

    return params


@njit
def convert_uniform_to_params(uniform, param_ranges):
    """Convert uniform random samples to parameter values"""
    sampled = convert_uniform_to_sampled_quantities(uniform, param_ranges)
    params = convert_sampled_quantities_to_params(sampled, param_ranges)
    return params


@njit
def package_params_for_ssa(params) -> tuple:
    """Set up reaction propensities in convenient form"""

    (
        k_on,
        k_off_1,
        k_off_2,
        km_unbound,
        km_act,
        km_rep,
        km_act_rep,
        kp,
        gamma_m,
        gamma_p,
    ) = params

    # Transcription rates are stored in a nested list
    # First layer is number of activators and second layer is number of repressors
    k_tx = np.array(
        [
            [km_unbound, km_rep, km_rep],
            [km_act, km_act_rep, 0.0],
            [km_act, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Promoter binding/unbinding are stored in index-able arrays, where the index
    # is the number of bound species.
    # NOTE: k_off_2 is multiplied by 2 because either of the two bound TFs can unbind.
    #     This stoichoimetry was erroneously neglected in the original implementation.
    k_ons = np.array([k_on, k_on, 0.0]).astype(np.float64)
    k_offs = np.array([0.0, k_off_1, 2 * k_off_2]).astype(np.float64)

    # This is the parameter tuple actually used for the propensity function
    # due to added efficiency
    return k_tx, kp, gamma_m, gamma_p, k_ons, k_offs


def convert_params_to_sampled_quantities(params, param_ranges=None, uniform=False):
    """Convert the parameters used in simulation to the quantities that were sampled
    initially. If `uniform=True`, rescale the value all the way back to the random
    number that was sampled from the uniform distribution on the inverval [0, 1)."""
    (
        k_on,
        k_off_1,
        k_off_2,
        km_unbound,
        km_act,
        km_rep,
        km_act_rep,
        kp,
        gamma_m,
        gamma_p,
    ) = params

    # Calculate quantities that were sampled
    log10_kd = np.log10(k_off_1 / k_on)
    kd_2_1_ratio = k_off_2 / k_off_1
    nlog10_km_rep_unbound_ratio = np.log10(km_unbound / km_rep)

    sampled_quantities = (
        log10_kd,
        kd_2_1_ratio,
        km_unbound,
        km_act,
        nlog10_km_rep_unbound_ratio,
        kp,
        gamma_m,
        gamma_p,
    )

    if uniform:
        if param_ranges is None:
            raise ValueError("param_ranges must be specified if uniform is True")
        return np.array(
            [
                _normalize(x, lo, hi)
                for x, (lo, hi) in zip(sampled_quantities, param_ranges)
            ]
        )
    else:
        return sampled_quantities


def _normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


@njit
def _rescale(x, xmin, xmax):
    return x * (xmax - xmin) + xmin


@njit
def convert_uniform_to_sampled_quantities(uniform, param_ranges):
    n = len(param_ranges)
    sampled = np.zeros(n, dtype=np.float64)
    for i, (lo, hi) in enumerate(param_ranges):
        sampled[i] = _rescale(uniform[i], lo, hi)
    return sampled


@njit
def draw_random_params(rg, param_ranges):
    """Make random draws for each quantity needed to define a parameter set"""
    uniform = np.array([rg.uniform(0.0, 1.0) for _ in param_ranges])
    params = convert_uniform_to_params(uniform, param_ranges)
    return rg, params


@njit
def draw_random_initial_protein(rg, m, poisson_mean):
    prot0 = rg.poisson(poisson_mean, m)
    return rg, prot0


@njit
def _population_from_proteins(
    proteins: np.ndarray, fill_value: int, m: int, n_species: int
):
    pop0 = np.full(n_species, fill_value, dtype=np.int64)
    for i, p in enumerate(proteins):
        pop0[m + i] = p
    return pop0


@njit
def draw_random_initial(rg, m, a, r, poisson_mean):
    m2 = 2 * m
    n_species = m2 + a + r
    pop0 = np.zeros(n_species).astype(np.int64)
    rg, pop0[m:m2] = draw_random_initial_protein(rg, m, poisson_mean)
    return rg, pop0


@njit
def draw_random_initial_and_params(rg, SAMPLING_RANGES, m, a, r, poisson_mean):
    rg, pop0 = draw_random_initial(rg, m, a, r, poisson_mean)
    rg, params = draw_random_params(rg, SAMPLING_RANGES)
    return rg, pop0, params


def make_matrices_for_ssa(n_components, activations, inhibitions):
    """
    Generates:
    (1) The activation matrix ``Am``, with zeros everywhere except for 1s at the
    indices i, j where the ith TF activates the jth promoter.
    (2) The inhibition matrix ``Rm``, defined similarly
    (3) The update matrix ``U`` for the system. ``U`` is a
    ``(4M + 3A + 3R + 1, 2M + A + R)`` matrix that describes the change in each
    species for each reaction.
        - ``M`` is the number of TFs in the network
        - ``A`` is the number of activating interactions
        - ``R`` is the number of repressive interactions

    Reactions (rows of the update matrix):
        Transcription of mRNA (0, M)
        Translation of mRNA (M, 2M)
        Degradation of mRNA (2M, 3M)
        Degradation of unbound TF (3M, 4M)
        Binding of activator to promoter (4M, 4M + A)
        Unbinding of activator from promoter (4M + A, 4M + 2A)
        Degradation of bound activator (4M + 2A, 4M + 3A)
        Binding of inhibitor to promoter (4M + 3A, 4M + 3A + R)
        Unbinding of inhibitor from promoter (4M + 3A + R, 4M + 3A + 2R)
        Degradation of bound inhibitor (4M + 3A + 2R, 4M + 3A + 3R)
        Null reaction - all zeros (4M + 3A + 3R, 4M + 3A + 3R + 1)

    Species (columns of the update matrix):
        mRNA (0, M)
        Free TFs (M, 2M)
        Activator-promoter complexes (2M, 2M + A)
        Inhibitor-promoter complexes (2M + A, 2M + A + R)

    """
    m = n_components
    a = len(activations)
    r = len(inhibitions)
    m2 = 2 * m
    m3 = 3 * m
    m4 = 4 * m
    m6 = 6 * m
    a2 = 2 * a
    a3 = 3 * a
    r2 = 2 * r
    r3 = 3 * r
    m4a = m4 + a
    m4a2 = m4 + a2
    m4a3 = m4 + a3
    m4a3r = m4a3 + r
    m4a3r2 = m4a3 + r2
    m4a3r3 = m4a3 + r3
    m2a = m2 + a
    m2ar = m2a + r

    # Activation matrix
    Am = np.zeros((m, m)).astype(np.int64)
    for left, right in activations:
        Am[left, right] = 1

    # INhibition matrix
    Rm = np.zeros((m, m)).astype(np.int64)
    for left, right in inhibitions:
        Rm[left, right] = 1

    # Update matrix
    U = np.zeros((m4a3r3 + 1, m2ar)).astype(np.int64)

    U[:m2, :m2] = np.eye(m2).astype(np.int64)  # transcription/translation
    U[m2:m4, :m2] = -np.eye(m2).astype(np.int64)  # mRNA/free TF degradation

    ## Reactions relating to activation
    for j, (left, right) in enumerate(activations):
        # Binding
        U[m4 + j, m + left] = -1
        U[m4 + j, m2 + j] = 1

        # Unbinding
        U[m4a + j, m + left] = 1
        U[m4a + j, m2 + j] = -1

    # Degradation
    U[m4a2:m4a3, m2:m2a] = -np.eye(a).astype(np.int64)

    ## Reactions relating to inhibition (repression)
    for k, (left, right) in enumerate(inhibitions):
        # Binding
        U[m4a3 + k, m + left] = -1
        U[m4a3 + k, m2a + k] = 1

        # Unbinding
        U[m4a3r + k, m + left] = 1
        U[m4a3r + k, m2a + k] = -1

    # Degradation
    U[m4a3r2:m4a3r3, m2a:] = -np.eye(r).astype(np.int64)

    return Am, Rm, U


@njit
def _sample_discrete(rg, probs, probs_sum):
    q = rg.uniform() * probs_sum
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return rg, i - 1


@njit
def _sum(ar):
    return ar.sum()


@njit
def _draw_time(rg, props_sum):
    return rg, rg.exponential(1 / props_sum)


@njit
def _sample_propensities(rg, propensities):
    """
    Draws a reaction and the time it took to do that reaction.
    """

    props_sum = _sum(propensities)

    # Bail if the sum of propensities is zero
    if props_sum == 0.0:
        rxn = -1
        time = -1.0

    # Compute time and draw reaction from propensities
    else:
        rg, time = _draw_time(rg, props_sum)
        rg, rxn = _sample_discrete(rg, propensities, props_sum)

    return rg, rxn, time


@njit(fastmath=True)
def add_to_zeros(m, at, vals):
    """Add values to an array of zeros at given indices. Equivalent to `np.zeros`
    followed by `np.add.at`."""
    out = np.zeros((m,)).astype(np.int64)
    for idx, v in zip(at, vals):
        out[idx] += v
    return out


@njit
def _index_arr2d(arr2d, idx0, idx1):
    """Index a 2D array with two 1D arrays of indices."""
    n = len(idx0)
    out = np.zeros((n,)).astype(np.float64)
    for i in range(n):
        out[i] = arr2d[idx0[i], idx1[i]]
    return out


@njit
def get_propensities(
    propensities,
    population,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    m,
    a,
    r,
    *ssa_params,
):
    """
    Returns an array of propensities for each reaction.
    For M TFs, and A activation interactions, and R repression interactions,
    there are 6M + 2A + 2R elementary reactions.
    The ordering and indexing of the reactions is as follows:
        - Transcription of mRNA (0, M)
        - Translation of mRNA (M, 2M)
        - Degradation of mRNA (2M, 3M)
        - Degradation of unbound TF (3M, 4M)
        - Binding of activator to promoter (4M, 4M + A)
        - Unbinding of activator from promoter (4M + A, 4M + 2A)
        - Degradation of bound activator (4M + 2A, 4M + 3A)
        - Binding of inhibitor to promoter (4M + 3A, 4M + 3A + R)
        - Unbinding of inhibitor from promoter (4M + 3A + R, 4M + 3A + 2R)
        - Degradation of bound inhibitor (4M + 3A + 2R, 4M + 3A + 3R)
    """
    (
        k_tx,
        kp,
        gamma_m,
        gamma_p,
        k_ons,
        k_offs,
    ) = ssa_params

    m2 = 2 * m
    m3 = m2 + m
    m4 = m3 + m
    m4a = m4 + a
    m4a2 = m4a + a
    m4a3 = m4a2 + a
    m4a3r = m4a3 + r
    m4a3r2 = m4a3r + r
    m4a3r3 = m4a3r2 + r

    # mRNA and protein for each TF + activators/repressors bound to each promoter
    m_s = population[:m]
    p_s = population[m:m2]
    ap_complexes = population[m2 : m2 + a]
    rp_complexes = population[m2 + a :]

    # Get the number of activators and repressors bound to each promoter
    a_bound = add_to_zeros(m, activations_right, ap_complexes)
    r_bound = add_to_zeros(m, inhibitions_right, rp_complexes)
    n_bound = a_bound + r_bound

    # Transcription
    propensities[:m] = _index_arr2d(k_tx, a_bound, r_bound)

    # Translation
    propensities[m:m2] = kp * m_s

    # mRNA degradation
    propensities[m2:m3] = gamma_m * m_s

    # protein degradation
    propensities[m3:m4] = gamma_p * p_s

    # Activator binding
    propensities[m4:m4a] = p_s[activations_left] * k_ons[n_bound[activations_right]]

    # Activator unbinding
    propensities[m4a:m4a2] = k_offs[ap_complexes]

    # Bound activator degradation
    propensities[m4a2:m4a3] = gamma_p * ap_complexes

    # inhibitor binding
    propensities[m4a3:m4a3r] = p_s[inhibitions_left] * k_ons[n_bound[inhibitions_right]]

    # inhibitor unbinding
    propensities[m4a3r:m4a3r2] = k_offs[rp_complexes]

    # Bound inhibitor degradation
    propensities[m4a3r2:m4a3r3] = gamma_p * rp_complexes

    return propensities


@njit
def _apply_event(population, U, event):
    """Apply an event to the population in-place."""
    population += U[event]


@njit
def take_gillespie_step(
    rg,
    t,
    population,
    event,
    propensities,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    m,
    a,
    r,
    *ssa_params,
):
    # draw the event and time step
    propensities = get_propensities(
        propensities,
        population,
        activations_left,
        activations_right,
        inhibitions_left,
        inhibitions_right,
        m,
        a,
        r,
        *ssa_params,
    )
    rg, event, dt = _sample_propensities(rg, propensities)

    # Skip to the end of the simulation
    if event == -1:
        t = np.inf
    else:
        # Increment time
        # If t exceeds the next time point, population isn't updated
        t += dt

    return rg, t, event


@njit
def gillespie_trajectory(
    rg,
    time_points,
    population_0,
    U,
    m,
    a,
    r,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    max_iter_per_timestep,
    *ssa_params,
):
    """ """

    # Initialize output
    nt = len(time_points)
    m4a3r3p1, m2ar = U.shape
    m4a3r3 = m4a3r3p1 - 1
    pop_out = -np.ones((nt, m2ar)).astype(np.int64)

    # Initialize and perform simulation
    propensities = np.zeros(m4a3r3).astype(np.float64)
    population = population_0.copy().astype(np.int64)
    pop_out[0] = population
    j = 0
    t = time_points[0]

    # First loop makes no changes (indexes the all-zero row of update matrix)
    event = m4a3r3
    while j < nt:
        n_iter = 0
        tj = time_points[j]
        while t <= tj:
            _apply_event(population, U, event)
            rg, t, event = take_gillespie_step(
                rg,
                t,
                population,
                event,
                propensities,
                activations_left,
                activations_right,
                inhibitions_left,
                inhibitions_right,
                m,
                a,
                r,
                *ssa_params,
            )
            n_iter += 1
            if n_iter > max_iter_per_timestep:
                pop_out[j:] = np.nan
                return rg, pop_out

        # Update the index (Be careful about types for Numba)
        new_j = j + np.searchsorted(time_points[j:], t)

        # Update the population
        pop_out[j:new_j] = population

        # Increment index
        j = new_j

    return rg, pop_out


@njit
def gillespie_trajectory_with_n_iter(
    rg,
    time_points,
    population_0,
    U,
    m,
    a,
    r,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    *ssa_params,
):
    """Same as gillespie_trajectory() but also keeps track of the number of iterations
    at each time point."""

    # Initialize output
    nt = len(time_points)
    m4a3r3p1, m2ar = U.shape
    m4a3r3 = m4a3r3p1 - 1
    pop_out = -np.ones((nt, m2ar)).astype(np.int64)

    # Initialize and perform simulation
    propensities = np.zeros(m4a3r3).astype(np.float64)
    population = population_0.copy().astype(np.int64)
    pop_out[0] = population
    j = 0
    t = time_points[0]

    n_iter_out = -np.ones(nt).astype(np.int64)
    n_iter = 0

    # First loop makes no changes (indexes the all-zero row of update matrix)
    event = m4a3r3
    while j < nt:
        tj = time_points[j]
        while t <= tj:
            _apply_event(population, U, event)
            rg, t, event = take_gillespie_step(
                rg,
                t,
                population,
                event,
                propensities,
                activations_left,
                activations_right,
                inhibitions_left,
                inhibitions_right,
                m,
                a,
                r,
                *ssa_params,
            )
            n_iter += 1

        # Update the index (Be careful about types for Numba)
        new_j = j + np.searchsorted(time_points[j:], t)

        # Update the population
        pop_out[j:new_j] = population
        n_iter_out[j:new_j] = n_iter

        # Increment index
        j = new_j

    return rg, pop_out, n_iter_out


@njit
def add_outer_1d_2d(arr1d, arr2d):
    """Perform outer addition of a 1D array to a 2D array. Equivalent to
    `np.add.outer`."""
    (k,) = arr1d.shape
    (m, n) = arr2d.shape
    out = np.zeros((k, m, n)).astype(arr1d.dtype)
    for i in range(k):
        out[i] = arr1d[i] + arr2d
    return out


def get_params_with_knockdown(
    params, which_knockdown: int, knockdown_coeff: float, m: int
):
    """Get the parameters to simulate knockdown of a single TF by a factor
    `knockdown_coeff` on the interval `[0.0, 1.0]` (0.0 indicating complete
    knockdown). The parameter `which_knockdown` indicates which TF to knock down.

    Knockdown is achieved by multiplying all transcription rates of the knocked
    down TF by `knockdown_coeff`. All other parameters are unchanged.
    """
    # Make an (m x n_params) array of parameters for each TF
    asymmetric_params = np.tile(params, (m, 1))

    # Multiply the transcription rates `k_unbound`, `k_act`, `k_rep`, and `k_act_rep`
    # by `knockdown_coeff` for the knocked down TF
    asymmetric_params[which_knockdown, 3:7] *= knockdown_coeff
    return asymmetric_params


def get_params_perturbed(
    params, sigma_params, m, param_ranges=None, rg: np.random.Generator = None
):
    if np.isclose(sigma_params, 0.0):
        return rg, np.tile(params, (m, 1))

    if param_ranges is None:
        param_ranges = SAMPLING_RANGES
    if rg is None:
        rg = np.random.default_rng()

    uniform_sampled_quantities = convert_params_to_sampled_quantities(
        params, param_ranges, uniform=True
    )
    n_sampled = len(uniform_sampled_quantities)
    perturbed_uniform_samples = np.zeros((m, n_sampled)).astype(np.float64)
    for j in range(n_sampled):
        # Sample from a normal distribution centered on the sampled value
        # (between 0 and 1) with a standard deviation of sigma_params
        loc = uniform_sampled_quantities[j]
        a = (0 - loc) / sigma_params
        b = (1 - loc) / sigma_params
        distribution = truncnorm(a, b, loc=loc, scale=sigma_params)
        perturbed_uniform_samples[:, j] = distribution.rvs(m, random_state=rg)

    # Convert perturbed uniform samples to parameter values
    perturbed_params = np.zeros((m, 10)).astype(np.float64)
    for i in range(m):
        perturbed_params[i] = convert_uniform_to_params(
            perturbed_uniform_samples[i], param_ranges
        )

    return rg, perturbed_params


def package_params_for_ssa_asymmetric(asymmetric_params):
    """Construct the parameter tuple to run the SSA with asymmetric TFs."""
    return tuple(
        np.array(args)
        for args in zip(*[package_params_for_ssa(p) for p in asymmetric_params])
    )


@njit
def _index_arr3d_range(arr3d, idx1, idx2):
    """Index a 3D array with two 1D arrays of indices and return a 1D array. The first
    element will be `arr3d[0, idx1[0], idx2[0]]`, etc.."""
    n = len(idx1)
    out = np.zeros((n,)).astype(np.float64)
    for i in range(n):
        out[i] = arr3d[i, idx1[i], idx2[i]]
    return out


@njit
def get_propensities_asymmetric(
    propensities,
    population,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    m,
    a,
    r,
    *ssa_params_asymmetric,
):
    """
    Returns an array of propensities for each reaction in a system of M TFs.

    This is the same as get_propensities() but each parameter has a different value
    for each TF (i.e. TFs are asymmetric with respect to their reaction rates).
    All the elements in `ssa_params_with_perturbation` have an additional leading axis
    of length M. For instance, where get_propensities() has a single value for
    `gamma_m`, this function takes the parameter `gamma_m_per` which is an array of
    length M.

    For M TFs, A activation interactions, and R repression interactions,
    there are 6M + 2A + 2R elementary reactions.
    The ordering and indexing of the reactions is as follows:
        - Transcription of mRNA (0, M)
        - Translation of mRNA (M, 2M)
        - Degradation of mRNA (2M, 3M)
        - Degradation of unbound TF (3M, 4M)
        - Binding of activator to promoter (4M, 4M + A)
        - Unbinding of activator from promoter (4M + A, 4M + 2A)
        - Degradation of bound activator (4M + 2A, 4M + 3A)
        - Binding of inhibitor to promoter (4M + 3A, 4M + 3A + R)
        - Unbinding of inhibitor from promoter (4M + 3A + R, 4M + 3A + 2R)
        - Degradation of bound inhibitor (4M + 3A + 2R, 4M + 3A + 3R)
    """
    (
        k_tx_per,
        kp_per,
        gamma_m_per,
        gamma_p_per,
        k_ons_per,
        k_offs_per,
    ) = ssa_params_asymmetric

    m2 = 2 * m
    m3 = m2 + m
    m4 = m3 + m
    m4a = m4 + a
    m4a2 = m4a + a
    m4a3 = m4a2 + a
    m4a3r = m4a3 + r
    m4a3r2 = m4a3r + r
    m4a3r3 = m4a3r2 + r

    # mRNA and protein for each TF + activators/repressors bound to each promoter
    m_s = population[:m]
    p_s = population[m:m2]
    ap_complexes = population[m2 : m2 + a]
    rp_complexes = population[m2 + a :]

    # Get the number of activators and repressors bound to each promoter
    a_bound = add_to_zeros(m, activations_right, ap_complexes)
    r_bound = add_to_zeros(m, inhibitions_right, rp_complexes)
    n_bound = a_bound + r_bound

    # Transcription
    propensities[:m] = _index_arr3d_range(k_tx_per, a_bound, r_bound)

    # Translation
    propensities[m:m2] = kp_per * m_s

    # mRNA degradation
    propensities[m2:m3] = gamma_m_per * m_s

    # protein degradation
    propensities[m3:m4] = gamma_p_per * p_s

    # Activator binding
    propensities[m4:m4a] = p_s[activations_left] * _index_arr2d(
        k_ons_per, activations_left, n_bound[activations_right]
    )

    # Activator unbinding
    propensities[m4a:m4a2] = _index_arr2d(k_offs_per, activations_left, ap_complexes)

    # Bound activator degradation
    propensities[m4a2:m4a3] = gamma_p_per[activations_left] * ap_complexes

    # inhibitor binding
    propensities[m4a3:m4a3r] = p_s[inhibitions_left] * _index_arr2d(
        k_ons_per, inhibitions_left, n_bound[inhibitions_right]
    )

    # inhibitor unbinding
    propensities[m4a3r:m4a3r2] = _index_arr2d(
        k_offs_per, inhibitions_left, rp_complexes
    )

    # Bound inhibitor degradation
    propensities[m4a3r2:m4a3r3] = gamma_p_per[inhibitions_left] * rp_complexes

    return propensities


@njit
def take_gillespie_step_asymmetric(
    rg,
    t,
    population,
    event,
    propensities,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    m,
    a,
    r,
    *ssa_params_asymmetric,
):
    # draw the event and time step
    propensities = get_propensities_asymmetric(
        propensities,
        population,
        activations_left,
        activations_right,
        inhibitions_left,
        inhibitions_right,
        m,
        a,
        r,
        *ssa_params_asymmetric,
    )
    rg, event, dt = _sample_propensities(rg, propensities)

    # Skip to the end of the simulation
    if event == -1:
        t = np.inf
    else:
        # Increment time
        # If t exceeds the next time point, population isn't updated
        t += dt

    return rg, t, event


@njit
def gillespie_trajectory_asymmetric(
    rg,
    time_points,
    population_0,
    U,
    m,
    a,
    r,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    max_iter_per_timestep,
    *ssa_params_asymmetric,
):
    """Same as gillespie_trajectory() but individual components may have different
    (asymmetric) values for each parameter."""

    # Initialize output
    nt = len(time_points)
    m4a3r3p1, m2ar = U.shape
    m4a3r3 = m4a3r3p1 - 1
    pop_out = -np.ones((nt, m2ar)).astype(np.int64)

    # Initialize and perform simulation
    propensities = np.zeros(m4a3r3).astype(np.float64)
    population = population_0.copy().astype(np.int64)
    pop_out[0] = population
    j = 0
    t = time_points[0]

    # First loop makes no changes (indexes the all-zero row of update matrix)
    event = m4a3r3
    while j < nt:
        n_iter = 0
        tj = time_points[j]
        while t <= tj:
            _apply_event(population, U, event)
            rg, t, event = take_gillespie_step_asymmetric(
                rg,
                t,
                population,
                event,
                propensities,
                activations_left,
                activations_right,
                inhibitions_left,
                inhibitions_right,
                m,
                a,
                r,
                *ssa_params_asymmetric,
            )
            n_iter += 1
            if n_iter > max_iter_per_timestep:
                pop_out[j:] = np.nan
                return rg, pop_out

        # Update the index (Be careful about types for Numba)
        new_j = j + np.searchsorted(time_points[j:], t)

        # Update the population
        pop_out[j:new_j] = population

        # Increment index
        j = new_j

    return rg, pop_out


class GillespieSSA:
    def __init__(
        self,
        seed,
        n_species,
        activation_mtx,
        inhibition_mtx,
        update_mtx,
        dt,
        nt,
        mean_protein_init,
        SAMPLING_RANGES,
        DEFAULT_PARAMS,
        max_iter_per_timestep=100_000_000,
    ):
        self.rg = np.random.default_rng(seed)

        self.init_mean = mean_protein_init

        self.SAMPLING_RANGES = SAMPLING_RANGES
        self.DEFAULT_PARAMS = DEFAULT_PARAMS

        # Compute some numbers for convenient indexing
        self.m = n_species
        self.a = activation_mtx.sum()
        self.r = inhibition_mtx.sum()
        self.m2 = 2 * self.m
        self.m3 = 3 * self.m
        self.m4 = 4 * self.m
        self.m6 = 6 * self.m
        self.a2 = 2 * self.a
        self.a3 = 3 * self.a
        self.r2 = 2 * self.r
        self.r3 = 3 * self.r
        self.m4a = self.m4 + self.a
        self.m4a2 = self.m4 + self.a2
        self.m4a3 = self.m4 + self.a3
        self.m4a3r = self.m4a3 + self.r
        self.m4a3r2 = self.m4a3 + self.r2
        self.m4a3r3 = self.m4a3 + self.r3
        self.m2a = self.m2 + self.a
        self.m2ar = self.m2a + self.r

        self.n_propensities = self.m4a3r3
        self.n_species = self.m2ar
        self.n_params = len(self.DEFAULT_PARAMS)

        # Get indices of TFs involved in the left- and right-hand side of each reaction
        self.activations_left, self.activations_right = activation_mtx.nonzero()
        self.inhibitions_left, self.inhibitions_right = inhibition_mtx.nonzero()

        # Specify a limit on the number of Gillespie iterations per timestep
        # This allows us to catch cases where the simulation is taking too long
        self.max_iter_per_timestep = max_iter_per_timestep

        self.U = update_mtx

        self.dt = dt
        self.nt = nt
        self.time_points = dt * np.arange(nt)

    def population_from_proteins(self, proteins: np.ndarray, fill_value: int = 0):
        """ """
        return _population_from_proteins(proteins, fill_value, self.m, self.n_species)

    def set_nt(self, nt: int) -> None:
        self.nt = nt
        self.time_points = self.dt * np.arange(nt)

    def set_dt(self, dt: float) -> None:
        self.dt = dt
        self.time_points = dt * np.arange(self.nt)

    def set_time_points(self, t: np.ndarray) -> None:
        self.dt = t[1] - t[0]
        self.nt = len(t)
        self.time_points = np.array(t, dtype=np.float64)

    def gillespie_trajectory(
        self,
        population_0,
        *ssa_params,
        seed: Optional[int] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        self.rg, y_t = gillespie_trajectory(
            self.rg,
            self.time_points,
            population_0,
            self.U,
            self.m,
            self.a,
            self.r,
            self.activations_left,
            self.activations_right,
            self.inhibitions_left,
            self.inhibitions_right,
            self.max_iter_per_timestep,
            *ssa_params,
        )
        if not maxiter_ok:
            y_t_isnan = np.isnan(y_t)
            if y_t_isnan.any():
                nanstep = np.where(y_t_isnan)[0][0]
                raise RuntimeError(
                    f"Number of Gillespie steps at time-step {nanstep} exceeded"
                    f" the maximum allowed number of {self.max_iter_per_timestep}."
                )
        return y_t

    def gillespie_n_iter(self, population_0, *ssa_params, seed: Optional[int] = None):
        """ """
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        self.rg, y_t, n_iter = gillespie_trajectory_with_n_iter(
            self.rg,
            self.time_points,
            population_0,
            self.U,
            self.m,
            self.a,
            self.r,
            self.activations_left,
            self.activations_right,
            self.inhibitions_left,
            self.inhibitions_right,
            *ssa_params,
        )
        return y_t, n_iter

    def gillespie_asymmetric(
        self,
        population_0,
        *ssa_params_asymmetric,
        seed: Optional[int] = None,
        maxiter_ok: bool = True,
    ):
        """Perform a Gillespie simulation with asymmetric TFs (different rate
        parameters for each)."""
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        self.rg, y_t = gillespie_trajectory_asymmetric(
            self.rg,
            self.time_points,
            population_0,
            self.U,
            self.m,
            self.a,
            self.r,
            self.activations_left,
            self.activations_right,
            self.inhibitions_left,
            self.inhibitions_right,
            self.max_iter_per_timestep,
            *ssa_params_asymmetric,
        )
        if not maxiter_ok:
            y_t_isnan = np.isnan(y_t)
            if y_t_isnan.any():
                nanstep = np.where(y_t_isnan)[0][0]
                raise RuntimeError(
                    f"Number of Gillespie steps at time-step {nanstep} exceeded"
                    f" the maximum allowed number of {self.max_iter_per_timestep}."
                )
        return y_t

    def get_params_with_knockdown(
        self, params, which_knockdown: int, knockdown_coeff: float
    ):
        """ """
        return get_params_with_knockdown(
            params, which_knockdown, knockdown_coeff, self.m
        )

    def get_params_with_perturbation(self, params, sigma_params):
        """ """
        self.rg, params_with_perturbation = get_params_perturbed(
            params, sigma_params, self.m, rg=self.rg
        )
        return params_with_perturbation

    def package_params_for_asymmetric_ssa(self, asymmetric_params):
        """ """
        return package_params_for_ssa_asymmetric(asymmetric_params)

    def run_asymmetric(
        self,
        pop0,
        params,
        seed: Optional[int] = None,
        nchunks: int = 1,
        maxiter_ok: bool = True,
    ):
        """ """
        pop0 = np.atleast_2d(pop0)
        if params.ndim == 2:
            params = params[None, ...]
        elif params.ndim != 3:
            raise ValueError(
                "Parameter sets must be a 2D array or a 3D array in which the first axis "
                "corresponds to the number of parameter sets."
            )
        n_init_states = pop0.shape[0]
        n_param_sets = params.shape[0]
        if n_init_states != n_param_sets:
            raise ValueError(
                f"Number of parameter sets ({n_param_sets}) and initial simulation "
                f"conditions ({n_init_states}) must match."
            )

        batch_size = n_init_states
        if batch_size == 1:
            pop0 = pop0[0]
            params = params[0]
            ssa_params = self.package_params_for_asymmetric_ssa(params)
            if nchunks == 1:
                y_t = self.gillespie_asymmetric(
                    pop0, *ssa_params, seed=seed, maxiter_ok=maxiter_ok
                )
            else:
                y_t = self.run_asymmetric_in_chunks(
                    pop0, ssa_params, seed=seed, nchunks=nchunks, maxiter_ok=maxiter_ok
                )
        else:
            if nchunks == 1:
                y_t = self.run_asymmetric_batch(
                    pop0, params, seed=seed, maxiter_ok=maxiter_ok
                )
            else:
                y_t = self.run_asymmetric_batch_in_chunks(
                    pop0, params, seed=seed, nchunks=nchunks, maxiter_ok=maxiter_ok
                )

        return y_t

    def run_asymmetric_batch(
        self,
        pop0s,
        param_sets,
        seed: Optional[int | Iterable[int]] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        n = len(pop0s)
        iter_seeds = self._random_seed_iterator(n, seed)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i, s in enumerate(iter_seeds):
            self.rg = np.random.default_rng(s)
            ssa_params = self.package_params_for_asymmetric_ssa(param_sets[i])
            y_ts[i] = self.gillespie_asymmetric(
                pop0s[i], *ssa_params, maxiter_ok=maxiter_ok
            )
        return y_ts

    def run_asymmetric_in_chunks(
        self,
        pop0,
        params,
        seed=None,
        nchunks=1,
        maxiter_ok=True,
    ):
        """ """
        if seed is not None:
            self.rg = np.random.default_rng(seed)

        # Get the number of time points in each chunk
        nt_chunk, chunk_mod = divmod(self.nt, nchunks)
        if chunk_mod > 0:
            nt_chunk += 1

        # Back up time parameters
        t_complete = self.time_points.copy()
        nt = self.nt

        # Initialize output
        y_t = np.zeros((nt, self.n_species), dtype=np.int64)
        pop0_chunk = pop0

        # Run in chunks
        for i in range(0, nt, nt_chunk):
            t_chunk = t_complete[i : i + nt_chunk + 1]
            self.set_time_points(t_chunk)
            pop_t_chunk = self.gillespie_asymmetric(
                pop0_chunk, *params, seed=seed, maxiter_ok=maxiter_ok
            )
            y_t[i : i + nt_chunk] = pop_t_chunk[:nt_chunk]
            pop0_chunk = pop_t_chunk[-1]

        self.set_time_points(t_complete)
        return y_t

    def run_asymmetric_batch_in_chunks(
        self,
        pop0s,
        param_sets,
        seed: Optional[int | Iterable[int]] = None,
        nchunks: int = 1,
        maxiter_ok: bool = True,
    ):
        """ """
        n = len(pop0s)
        iter_seeds = self._random_seed_iterator(n, seed)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i, s in enumerate(iter_seeds):
            self.rg = np.random.default_rng(s)
            ssa_params = self.package_params_for_asymmetric_ssa(param_sets[i])
            y_ts[i] = self.run_asymmetric_in_chunks(
                pop0s[i], ssa_params, seed=s, nchunks=nchunks, maxiter_ok=maxiter_ok
            )
        return y_ts

    def get_ssa_params_with_knockdown(
        self, params, which_knockdown: int, knockdown_coeff: float
    ):
        """ """
        params_with_knockdown = self.get_params_with_knockdown(
            params, which_knockdown, knockdown_coeff
        )
        return self.package_params_for_asymmetric_ssa(params_with_knockdown)

    def get_ssa_params_with_perturbation(self, params, sigma_params):
        """ """
        params_with_perturbation = self.get_params_with_perturbation(
            params, sigma_params
        )
        return self.package_params_for_asymmetric_ssa(params_with_perturbation)

    def run_with_params_and_knockdown(
        self,
        pop0,
        params,
        which_knockdown: int,
        knockdown_coeff: float,
        seed: Optional[int] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        ssa_params_kd = self.get_ssa_params_with_knockdown(
            params, which_knockdown, knockdown_coeff
        )
        return self.gillespie_asymmetric(
            pop0, *ssa_params_kd, seed=seed, maxiter_ok=maxiter_ok
        )

    def run_with_params_and_perturbation(
        self,
        pop0,
        params,
        sigma_params,
        seed: Optional[int] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        ssa_params_ptb = self.get_ssa_params_with_perturbation(params, sigma_params)
        return self.gillespie_asymmetric(
            pop0, *ssa_params_ptb, seed=seed, maxiter_ok=maxiter_ok
        )

    def run_with_params_in_chunks_with_knockdown(
        self,
        pop0,
        params,
        which_knockdown: int,
        knockdown_coeff: float,
        nchunks=1,
        seed=None,
        maxiter_ok=True,
    ):
        """Run an SSA with knockdown of a TF, splitting time into nchunks chunks. This
        allows the simulation to be interrupted by KeyboardInterrupt/SIGINT."""
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        ssa_params_kd = self.get_ssa_params_with_knockdown(
            params, which_knockdown, knockdown_coeff
        )
        time_points = self.time_points
        nt_chunk, chunk_mod = divmod(self.nt, nchunks)
        if chunk_mod > 0:
            nt_chunk += 1
        pop_t = np.zeros((self.nt, self.n_species), dtype=np.int64)
        pop0_chunk = pop0
        for i in range(0, self.nt, nt_chunk):
            t_chunk = time_points[i : i + nt_chunk + 1]
            self.set_time_points(t_chunk)
            pop_t_chunk = self.gillespie_asymmetric(
                pop0_chunk, *ssa_params_kd, maxiter_ok=maxiter_ok
            )
            pop_t[i : i + nt_chunk] = pop_t_chunk[:nt_chunk]
            pop0_chunk = pop_t_chunk[-1]
        return pop_t

    def run_with_params_in_chunks_with_perturbation(
        self, pop0, params, sigma_params, nchunks=1, seed=None, maxiter_ok=True
    ):
        """Run a parameter-perturbed SSA, splitting time into nchunks chunks. This
        allows the simulation to be interrupted by KeyboardInterrupt/SIGINT."""
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        ssa_params_ptb = self.get_ssa_params_with_perturbation(params, sigma_params)
        time_points = self.time_points
        nt_chunk, chunk_mod = divmod(self.nt, nchunks)
        if chunk_mod > 0:
            nt_chunk += 1
        pop_t = np.zeros((self.nt, self.n_species), dtype=np.int64)
        pop0_chunk = pop0
        for i in range(0, self.nt, nt_chunk):
            t_chunk = time_points[i : i + nt_chunk + 1]
            self.set_time_points(t_chunk)
            pop_t_chunk = self.gillespie_asymmetric(
                pop0_chunk, *ssa_params_ptb, maxiter_ok=maxiter_ok
            )
            pop_t[i : i + nt_chunk] = pop_t_chunk[:nt_chunk]
            pop0_chunk = pop_t_chunk[-1]
        return pop_t

    def run_random_sample_with_knockdown(
        self,
        which_knockdown: int,
        knockdown_coeff: float,
        seed: Optional[int] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        if seed is not None:
            self.rg = np.random.default_rng(seed)

        self.rg, params = draw_random_params(self.rg, SAMPLING_RANGES)

        # Use an all-zero initial population
        pop0 = np.zeros((self.n_species,)).astype(np.int64)

        ssa_params_kd = self.get_ssa_params_with_knockdown(
            params, which_knockdown, knockdown_coeff
        )
        y_t = self.gillespie_asymmetric(
            pop0, *ssa_params_kd, seed=None, maxiter_ok=maxiter_ok
        )
        return pop0, params, y_t

    def run_random_sample_with_perturbation(
        self, sigma_params: float, seed: Optional[int] = None, maxiter_ok: bool = True
    ):
        """ """
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        pop0, params = self.draw_random_initial_and_params()
        params_ptb = self.get_params_with_perturbation(params, sigma_params)
        ssa_params_ptb = self.package_params_for_asymmetric_ssa(params_ptb)
        y_t = self.gillespie_asymmetric(
            pop0, *ssa_params_ptb, seed=None, maxiter_ok=maxiter_ok
        )
        return pop0, params_ptb, y_t

    def package_params_for_ssa(self, params):
        """ """
        return package_params_for_ssa(params)

    def draw_random_initial(self, init_mean=None):
        """ """
        init_mean = self.init_mean if init_mean is None else init_mean
        self.rg, pop0 = draw_random_initial(self.rg, self.m, self.a, self.r, init_mean)
        return pop0

    def draw_random_initial_and_params(self, init_mean=None):
        init_mean = self.init_mean if init_mean is None else init_mean
        self.rg, pop0, params = draw_random_initial_and_params(
            self.rg, self.SAMPLING_RANGES, self.m, self.a, self.r, init_mean
        )
        return pop0, params

    def _random_seed_iterator(
        self,
        n: int,
        seed_or_seeds: Optional[int | Iterable] = None,
    ):
        if seed_or_seeds is None:
            iter_seeds = repeat(None, n)
        elif isinstance(seed_or_seeds, int):
            iter_seeds = islice(enumerate(seed_or_seeds), n)
        elif isinstance(seed_or_seeds, Iterable):
            iter_seeds = islice(seed_or_seeds, n)
        else:
            raise TypeError("seed_or_seeds must be None, int, or Iterable")
        return iter_seeds

    def run_with_params(
        self, pop0, params, seed: Optional[int] = None, maxiter_ok: bool = True
    ):
        """ """
        ssa_params = self.package_params_for_ssa(params)
        return self.gillespie_trajectory(
            pop0, *ssa_params, seed=seed, maxiter_ok=maxiter_ok
        )

    def run_with_params_in_chunks(
        self, pop0, params, nchunks=1, seed=None, maxiter_ok=True
    ):
        """Run the simulation, splitting time into nchunks chunks. This allows the
        simulation to be interrupted by KeyboardInterrupt/SIGINT. Otherwise,
        Numba-compiled code will not respond to signals other than SIGKILL."""
        ssa_params = self.package_params_for_ssa(params)
        time_points = self.time_points
        nt_chunk, chunk_mod = divmod(self.nt, nchunks)
        if chunk_mod > 0:
            nt_chunk += 1
        pop_t = np.zeros((self.nt, self.n_species), dtype=np.int64)
        pop0_chunk = pop0
        for i in range(0, self.nt, nt_chunk):
            t_chunk = time_points[i : i + nt_chunk + 1]
            self.set_time_points(t_chunk)
            pop_t_chunk = self.gillespie_trajectory(
                pop0_chunk, *ssa_params, seed=seed, maxiter_ok=maxiter_ok
            )
            pop_t[i : i + nt_chunk] = pop_t_chunk[:nt_chunk]
            pop0_chunk = pop_t_chunk[-1]
        return pop_t

    def run_random_sample(self, seed: Optional[int] = None, maxiter_ok: bool = True):
        """ """
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        pop0, params = self.draw_random_initial_and_params()
        ssa_params = self.package_params_for_ssa(params)
        y_t = self.gillespie_trajectory(
            pop0, *ssa_params, seed=None, maxiter_ok=maxiter_ok
        )
        return pop0, params, y_t

    def run_batch(
        self,
        n: int,
        seed: Optional[int | Iterable[int]] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        iter_seeds = self._random_seed_iterator(n, seed)
        pop0s = np.zeros((n, self.n_species)).astype(np.int64)
        param_sets = np.zeros((n, self.n_params)).astype(np.float64)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i, s in enumerate(iter_seeds):
            self.rg = np.random.default_rng(s)
            pop0, params, y_t = self.run_random_sample(seed=None, maxiter_ok=maxiter_ok)
            pop0s[i] = pop0
            param_sets[i] = params
            y_ts[i] = y_t

        return pop0s, param_sets, y_ts

    def run_batch_with_params(
        self,
        pop0,
        params,
        n,
        seed: Optional[int | Iterable[int]] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        iter_seeds = self._random_seed_iterator(n, seed)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i, s in enumerate(iter_seeds):
            y_ts[i] = self.run_with_params(pop0, params, seed=s, maxiter_ok=maxiter_ok)
        return y_ts

    def run_batch_with_param_sets(
        self,
        pop0s,
        param_sets,
        seed: Optional[int | Iterable[int]] = None,
        maxiter_ok: bool = True,
    ):
        """ """
        n = len(pop0s)
        iter_seeds = self._random_seed_iterator(n, seed)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i, s in enumerate(iter_seeds):
            y_ts[i] = self.run_with_params(
                pop0s[i], param_sets[i], seed=s, maxiter_ok=maxiter_ok
            )
        return y_ts

    def run_n_iter_with_params(self, pop0, params, seed=None):
        """ """
        ssa_params = self.package_params_for_ssa(params)
        return self.gillespie_n_iter(pop0, *ssa_params, seed=seed)

    def run_n_iter_random(self, seed: Optional[int] = None):
        """ """
        if seed is not None:
            self.rg = np.random.default_rng(seed)
        pop0, params = self.draw_random_initial_and_params()
        ssa_params = self.package_params_for_ssa(params)
        y_t, n_iter = self.gillespie_n_iter(pop0, *ssa_params)
        return pop0, params, y_t, n_iter

    def run_n_iter_batch(self, n, seed: Optional[int | Iterable[int]] = None):
        """ """
        iter_seeds = self._random_seed_iterator(n, seed)
        pop0s = np.zeros((n, self.n_species)).astype(np.int64)
        param_sets = np.zeros((n, self.n_params)).astype(np.float64)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        n_iters = np.zeros((n, self.nt)).astype(np.int64)
        for i, s in enumerate(iter_seeds):
            pop0, params, y_t, n_iter = self.run_n_iter_random(seed=seed)
            pop0s[i] = pop0
            param_sets[i] = params
            y_ts[i] = y_t
            n_iters[i] = n_iter

        return pop0s, param_sets, y_ts, n_iters
