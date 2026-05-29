"""
physics.py  (week1-system-calibration)
--------------------------------------
Constants and closed-form relations for the Brownian analysis. Imported by the
analysis scripts so the numbers live in one place.

Stokes-Einstein:   D = k_B T / (6 pi eta r)      [m^2/s]
MSD (2D proj.):     <r^2> = 4 D t                  [m^2]
"""

K_B = 1.380649e-23   # Boltzmann constant [J/K]


def water_viscosity_cP(T_C):
    """Dynamic viscosity of liquid water [cP = mPa.s] vs temperature [degC].

    Standard empirical fit (ref. mu_20 = 1.0020 cP), valid ~0-100 degC:
        log10(mu/mu_20) = (20 - T)/(T + 96)
                          * (1.2378 - 1.303e-3 dT + 3.06e-6 dT^2 + 2.55e-8 dT^3)
    with dT = 20 - T.  Sanity points: 5 C -> 1.52, 15 C -> 1.14,
    20 C -> 1.00, 25 C -> 0.89 cP.
    """
    mu20 = 1.0020
    dT = 20.0 - T_C
    expo = (dT) / (T_C + 96.0) * (
        1.2378 - 1.303e-3 * dT + 3.06e-6 * dT**2 + 2.55e-8 * dT**3)
    return mu20 * 10.0 ** expo


def water_viscosity_Pa_s(T_C):
    """Same, in SI [Pa.s].  1 cP = 1e-3 Pa.s."""
    return water_viscosity_cP(T_C) * 1e-3


def stokes_einstein_D(T_C, r_m, eta_Pa_s=None):
    """Predicted diffusion coefficient D [m^2/s] for radius r_m [m] at T_C [degC]."""
    eta = water_viscosity_Pa_s(T_C) if eta_Pa_s is None else eta_Pa_s
    return K_B * (T_C + 273.15) / (6.0 * 3.141592653589793 * eta * r_m)


def kB_from_D_vs_invr_slope(slope_m3_s, T_C, eta_Pa_s=None):
    """Recover k_B from the slope of a D-vs-(1/r) line.

    D = (k_B T / 6 pi eta) * (1/r)  =>  slope = k_B T / (6 pi eta)
       =>  k_B = 6 pi eta * slope / T.
    `slope_m3_s` is dD/d(1/r) in SI [m^2/s per 1/m = m^3/s].
    """
    eta = water_viscosity_Pa_s(T_C) if eta_Pa_s is None else eta_Pa_s
    return 6.0 * 3.141592653589793 * eta * slope_m3_s / (T_C + 273.15)
