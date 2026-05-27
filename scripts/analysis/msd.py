"""
MSD (Mean Squared Displacement) computation and diffusion coefficient fitting.

Conventions:
- Positions in micrometres (µm), time in seconds (s)
- MSD = <r²(τ)> = 2d·D·τ for normal (Fickian) diffusion
- d = 2 for 2D tracking; set DIM below accordingly
- Fit range: exclude very short lags (localisation noise dominated) and
  very long lags (poor statistics). Typically fit τ ≤ T_total / 4.

Outputs:
- D   : diffusion coefficient [µm²/s]
- σ_D : uncertainty from fit covariance
- α   : anomalous diffusion exponent (α=1 → normal diffusion)
"""

import numpy as np
from scipy.optimize import curve_fit

DIM = 2  # spatial dimensions used in tracking


def compute_msd_single(xy: np.ndarray, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-averaged MSD for a single trajectory.

    Parameters
    ----------
    xy       : (N, 2) array of (x, y) positions in µm
    max_lag  : maximum lag in frames to compute; defaults to N//4

    Returns
    -------
    lags : (M,) int array of lag values in frames
    msd  : (M,) float array of MSD values in µm²
    """
    N = len(xy)
    if max_lag is None:
        max_lag = N // 4
    lags = np.arange(1, max_lag + 1)
    msd = np.array([
        np.mean(np.sum((xy[lag:] - xy[:-lag]) ** 2, axis=1))
        for lag in lags
    ])
    return lags, msd


def fit_diffusion(tau: np.ndarray, msd: np.ndarray, dim: int = DIM
                 ) -> tuple[float, float, float, float]:
    """
    Fit MSD = 2*dim*D*tau^alpha.

    Returns
    -------
    D, sigma_D, alpha, sigma_alpha
    """
    def model(t, D, alpha):
        return 2 * dim * D * t ** alpha

    popt, pcov = curve_fit(model, tau, msd, p0=[1.0, 1.0], maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    D, alpha = popt
    sigma_D, sigma_alpha = perr
    return float(D), float(sigma_D), float(alpha), float(sigma_alpha)


def stokes_einstein_D(T_K: float, eta: float, r_m: float) -> float:
    """
    Theoretical diffusion coefficient from Stokes-Einstein:
        D = k_B * T / (6 * pi * eta * r)

    Parameters
    ----------
    T_K  : temperature in Kelvin
    eta  : dynamic viscosity of solvent in Pa·s  (water at 20°C ≈ 1.002e-3)
    r_m  : particle radius in metres

    Returns
    -------
    D in m²/s  — convert to µm²/s by multiplying by 1e12
    """
    k_B = 1.380649e-23  # J/K
    return k_B * T_K / (6 * np.pi * eta * r_m)
