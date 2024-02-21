from datetime import datetime
import h5py
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
import pandas as pd

from tf_network import binomial9_kernel
from numba import prange, njit


@njit(parallel=True)
def binomial9_filter_parallel(x):
    """Apply a 9-point binomial filter over the second axis of a 2D array."""
    m, n = x.shape
    y = -np.ones((m, n - 8), dtype=np.float64)
    for i in prange(m):
        y[i] = binomial9_kernel(x[i])[4:-4]
    return y


def main(
    permutation_data_hdf: Path,
    dt_secs: float = 20.0,
    n_freq: int = 250,
    which_protein: int = 0,
    acf_threshold: float = -0.4,
    cmap: str = "viridis",
    save: bool = False,
    save_dir: Path = Path("results"),
    suffix: str = "",
    fmt: str = "png",
    dpi: int = 300,
):
    # Read in the data
    metadata = pd.read_hdf(permutation_data_hdf, key="metadata")
    with h5py.File(permutation_data_hdf, "r") as f:
        y_ts = f["y_ts"][:, :, which_protein]
        acf_minima = f["acf_minima"][:]

    # Filter with a 9-point binomial filter
    y_ts = binomial9_filter_parallel(y_ts)
    nt = y_ts.shape[1]

    # Compute the power spectral density
    sigma_params = metadata["sigma_param"].unique()
    log_sigma_min = np.log10(sigma_params.min())
    log_sigma_max = np.log10(sigma_params.max())
    n_sigma = len(sigma_params)
    psd = []
    print(f"Computing PSD for {len(y_ts)} total samples (n_sigma={n_sigma})")
    for i, sigma_param in enumerate(sigma_params):
        replicate_idx = metadata["sigma_param"] == sigma_param
        y_t = y_ts[replicate_idx]
        y_t = y_t - y_t.mean(axis=-1, keepdims=True)

        # Compute power spectrum using FFT
        sigma_fft = np.fft.rfft(y_t, axis=-1)
        sigma_power = np.abs(sigma_fft**2)

        # Normalize by total power. A signal with no power at any frequency will be
        # assumed to have a flat autocorrelation (FFT is a Dirac delta centered at 0)
        sigma_psd = np.zeros_like(sigma_power)
        sigma_psd[:, 0] = 1.0
        sigma_power_tot = sigma_power.sum(axis=-1, keepdims=True)
        where_nz = (sigma_power_tot > 0).ravel()
        sigma_psd[where_nz] = sigma_power[where_nz] / sigma_power_tot[where_nz]

        psd.append(sigma_psd.mean(axis=0)[1 : n_freq + 1])

    freqs = np.fft.rfftfreq(nt, d=dt_secs)[1 : n_freq + 1]
    freqs_per_hour = freqs * 3600
    psd = np.array(psd).T
    log_psd = np.log10(psd)

    # Plot the % oscillating vs sigma_param
    n_reps = metadata["replicate"].max() + 1
    pct_osc = -np.ones((n_sigma, n_reps), dtype=np.float64)
    acf_min = -np.ones((n_sigma, n_reps), dtype=np.float64)
    for i, sigma_param in enumerate(sigma_params):
        replicate_idx = metadata["sigma_param"] == sigma_param
        pct_osc[i] = acf_minima[replicate_idx] < acf_threshold
        acf_min[i] = acf_minima[replicate_idx]

    # Compute mean and spread
    pct_osc_mean = pct_osc.mean(axis=-1)
    pct_osc_sem = pct_osc.std(axis=-1) / np.sqrt(n_reps)
    acf_min_mean = acf_min.mean(axis=-1)
    acf_min_ci90_lo = np.quantile(acf_min, 0.05, axis=-1)
    acf_min_ci90_hi = np.quantile(acf_min, 0.95, axis=-1)

    # Construct figure
    from matplotlib.gridspec import GridSpec

    hr = 1.2, 4.0
    wr = 4.42, 0.58
    fig = plt.figure(figsize=(4, 4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=hr, width_ratios=wr)

    # Plot the % oscillating vs sigma_param
    # fig, ax = plt.subplots(figsize=(5, 3))
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sigma_params, pct_osc_mean, c="k", lw=1)
    ax1.fill_between(
        sigma_params,
        pct_osc_mean - pct_osc_sem,
        pct_osc_mean + pct_osc_sem,
        color="k",
        alpha=0.3,
    )
    # Format x-axis
    ax1.set_xlim(sigma_params.min(), sigma_params.max())
    ax1.set_xscale("log")
    # ax1.set_xlabel(r"$\sigma_\mathrm{param}$")
    ax1.set_ylabel(r"Frequency (hour$^{-1}$)")
    ax1.set_ylabel("P(oscillation)")

    # Make some space for the colorbar
    ax_blank = fig.add_subplot(gs[0, 1])
    ax_blank.axis("off")

    # Plot power spectral density
    ax2 = fig.add_subplot(gs[1, :])
    colormap = colormaps.get_cmap(cmap)
    X, Y = np.meshgrid(sigma_params, freqs_per_hour)
    im = ax2.pcolormesh(X, Y, psd, cmap=colormap, norm=LogNorm())

    # ax2.imshow(
    #     log_psd,
    #     cmap=colormap,
    #     aspect="auto",
    #     origin="lower",
    #     extent=(
    #         log_sigma_min,
    #         log_sigma_max,
    #         freqs_per_hour.min(),
    #         freqs_per_hour.max(),
    #     ),
    # )
    # ax2.set_xlabel(r"Log$_{10}[\sigma_\mathrm{param}]$")

    # Format x-axis
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\sigma_\mathrm{param}$")
    ax2.set_ylabel(r"Frequency (hour$^{-1}$)")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2, pad=0.01, label="PSD")

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = save_dir.joinpath(
            f"{today}_perturbation_oscillation_and_PSD{suffix}.{fmt}"
        )
        print(f"Writing to: {fpath.resolve().absolute()}")
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")

    ...

    # # Plot the % oscillating vs sigma_param
    # n_reps = metadata["replicate"].max() + 1
    # pct_osc = -np.ones((n_sigma, n_reps), dtype=np.float64)
    # acf_min = -np.ones((n_sigma, n_reps), dtype=np.float64)
    # for i, sigma_param in enumerate(sigma_params):
    #     replicate_idx = metadata["sigma_param"] == sigma_param
    #     pct_osc[i] = acf_minima[replicate_idx] < acf_threshold
    #     acf_min[i] = acf_minima[replicate_idx]

    # # Compute mean and spread
    # pct_osc_mean = pct_osc.mean(axis=-1)
    # pct_osc_sem = pct_osc.std(axis=-1) / np.sqrt(n_reps)
    # acf_min_mean = acf_min.mean(axis=-1)
    # acf_min_ci90_lo = np.quantile(acf_min, 0.05, axis=-1)
    # acf_min_ci90_hi = np.quantile(acf_min, 0.95, axis=-1)

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(sigma_params, pct_osc_mean, c="k", lw=1)
    # ax.fill_between(
    #     sigma_params,
    #     pct_osc_mean - pct_osc_sem,
    #     pct_osc_mean + pct_osc_sem,
    #     color="k",
    #     alpha=0.3,
    # )
    # ax.set_xlabel(r"$\sigma_\mathrm{param}$")
    # ax.set_xscale("log")
    # ax.set_ylabel("% oscillation")

    # if save:
    #     today = datetime.today().strftime("%y%m%d")
    #     fpath = save_dir.joinpath(f"{today}_perturbation_pct_osc{suffix}.{fmt}")
    #     print(f"Writing to: {fpath.resolve().absolute()}")
    #     fig.savefig(fpath, dpi=dpi, bbox_inches="tight")

    ...

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(sigma_params, acf_min_mean, c="k", lw=1)
    ax.fill_between(
        sigma_params,
        acf_min_ci90_lo,
        acf_min_ci90_hi,
        color="k",
        alpha=0.2,
        label="90% CI",
    )
    ax.set_xlabel(r"$\sigma_\mathrm{param}$")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\mathrm{ACF}_\mathrm{min}$")
    ax.legend()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = save_dir.joinpath(f"{today}_perturbation_acf_min{suffix}.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    permutation_data_hdf = Path(
        f"data/oscillation/240118_5TF_FTO3_perturbations/240118_perturbation_data_2.hdf5"
    )
    today = datetime.today().strftime("%y%m%d")
    save_dir = Path(f"figures/oscillation/{today}_FTO3_perturbations")
    save_dir.mkdir(exist_ok=True)

    main(
        permutation_data_hdf=permutation_data_hdf,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
