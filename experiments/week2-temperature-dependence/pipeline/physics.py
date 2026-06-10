"""
physics.py  (pipeline)
----------------------
Constants + closed-form relations (self-contained; no dependency on the old
scripts/physics.py). SI unless noted.

Stokes-Einstein:  D = k_B T / (6 pi eta r)
MSD (2D proj.):   <r^2> = 4 D t
"""

import numpy as np

K_B = 1.380649e-23      # accepted Boltzmann constant [J/K]
G = 9.80665             # gravity [m/s^2]


def water_viscosity_cP(T_C):
    """Dynamic viscosity of water [cP]; empirical fit (mu_20 = 1.0020 cP).
    Sanity: 5C->1.52, 15C->1.14, 20C->1.00, 25C->0.89 cP."""
    dT = 20.0 - T_C
    expo = dT / (T_C + 96.0) * (1.2378 - 1.303e-3 * dT + 3.06e-6 * dT ** 2
                                + 2.55e-8 * dT ** 3)
    return 1.0020 * 10.0 ** expo


def water_viscosity_Pa_s(T_C):
    return water_viscosity_cP(T_C) * 1e-3


def stokes_einstein_D(T_C, r_m, eta_Pa_s=None):
    eta = water_viscosity_Pa_s(T_C) if eta_Pa_s is None else eta_Pa_s
    return K_B * (T_C + 273.15) / (6.0 * np.pi * eta * r_m)


def kB_per_bead(D_um2_s, r_um, T_C, eta_Pa_s=None):
    """k_B,i = 6 pi eta r D / T from one bead's (D, r), in SI [J/K]."""
    eta = water_viscosity_Pa_s(T_C) if eta_Pa_s is None else eta_Pa_s
    r_m = np.asarray(r_um) * 1e-6
    D_m2 = np.asarray(D_um2_s) * 1e-12
    return 6.0 * np.pi * eta * r_m * D_m2 / (T_C + 273.15)


def kB_prefactor(T_C, eta_Pa_s=None):
    """k_B = prefactor * slope, where slope = dD/d(1/r) in SI [m^3/s].
    Also: k_B,i = prefactor * (D_i[m^2/s] * r_i[m]) ... with the um units folded
    in by callers. Returns 6 pi eta / T."""
    eta = water_viscosity_Pa_s(T_C) if eta_Pa_s is None else eta_Pa_s
    return 6.0 * np.pi * eta / (T_C + 273.15)


def sediment_r_star_um(T_C, delta_rho):
    """Radius [um] where the gravitational length equals the radius -- the
    free-diffusion boundary. Depends only on |delta_rho| (bead-fluid density
    difference), NOT on k_B input, so the cut is not circular.

    For POLYETHYLENE (rho ~ 0.91-0.96 g/cm^3 < water) the bead is BUOYANT and
    rises to the TOP coverslip; a denser bead would sink to the bottom. Either
    way a wall-pinned bead sub-diffuses (D below free SE) -> the SAME size-
    dependent bias, just at the opposite wall. r* is set by |delta_rho|."""
    kT = K_B * (T_C + 273.15)
    return (kT / ((4.0 / 3.0) * np.pi * abs(delta_rho) * G)) ** 0.25 * 1e6
