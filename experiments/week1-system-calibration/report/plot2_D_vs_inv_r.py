#!/usr/bin/env python3
"""
plot2_D_vs_inv_r.py
===================
Report figure 2 -- Diffusion coefficient vs inverse radius (Stokes-Einstein).

Goal
----
Take the per-bead diffusion coefficients D measured in plot 1 (one D per bead,
read from msd.csv) and the bead radii r (radius.csv) and show that they obey
the Stokes-Einstein relation

        D = (k_B T) / (6 pi eta r)        =>   D = slope * (1/r),
        slope = k_B T / (6 pi eta).

Plotting D against 1/r should give a straight line through the origin whose
slope fixes k_B:

        k_B = slope * 6 pi eta / T.

Physics caveat handled here
---------------------------
The polyethylene spheres are buoyant, so the largest beads drift up and graze
the top coverslip; wall drag suppresses their D below the free-diffusion line.
Those beads (radius above a sedimentation scale r*) are drawn in grey and kept
out of the fit, exactly as a free-diffusion analysis should. The line is fit to
the free beads only, through the origin.

This script is self-contained and does not reuse the pipeline / archive code.

Usage
-----
    python plot2_D_vs_inv_r.py                 # run3, T=25 C, eta from water
    python plot2_D_vs_inv_r.py --run run4 --T 25
    python plot2_D_vs_inv_r.py --no-cut        # fit every clean bead
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# Reuse the run-resolving / label-schema helpers from the plot1 script so every
# run (root-level run3 labels or pipeline/ keep-flags) loads the same way.
import plot1_msd_vs_lag as p1


# --------------------------------------------------------------------------- #
# Constants / paths
# --------------------------------------------------------------------------- #
K_B_ACCEPTED = 1.380649e-23   # CODATA exact [J/K]
G = 9.80665                   # gravity [m/s^2]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MEAS = os.path.join(ROOT, "measurements")

CLEAN_LABELS = {"perfect", "singlet"}


# --------------------------------------------------------------------------- #
# Physics helpers
# --------------------------------------------------------------------------- #
def water_viscosity_cP(T_C):
    """Dynamic viscosity of water [cP], empirical fit (1.0020 cP at 20 C)."""
    dT = 20.0 - T_C
    expo = dT / (T_C + 96.0) * (1.2378 - 1.303e-3 * dT + 3.06e-6 * dT ** 2
                                + 2.55e-8 * dT ** 3)
    return 1.0020 * 10.0 ** expo


def sediment_r_star_um(T_C, delta_rho):
    """Radius [um] where the gravitational length ~ the bead radius: the
    free/wall-bound boundary. Set by |bead-fluid density mismatch| only, so it
    is not circular with k_B."""
    kT = K_B_ACCEPTED * (T_C + 273.15)
    return (kT / ((4.0 / 3.0) * np.pi * abs(delta_rho) * G)) ** 0.25 * 1e6


def kB_from_slope(slope_um3_s, T_C, eta_Pa_s):
    """slope = dD/d(1/r) in [um^2/s * um] = um^3/s  ->  k_B [J/K]."""
    slope_SI = slope_um3_s * 1e-18                # um^3/s -> m^3/s
    return 6.0 * np.pi * eta_Pa_s * slope_SI / (T_C + 273.15)


def kB_per_bead(D_um2_s, r_um, T_C, eta_Pa_s):
    """k_B,i = 6 pi eta r D / T from a single bead, SI [J/K]."""
    return (6.0 * np.pi * eta_Pa_s * (r_um * 1e-6) * (D_um2_s * 1e-12)
            / (T_C + 273.15))


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_through_origin(x, y, yerr):
    """Weighted least-squares slope of y = m x (intercept forced to 0).

    Weights are 1/yerr^2; the returned slope error is rescaled by the reduced
    chi-square so it reflects the actual scatter, not just the input D errors.
    """
    w = 1.0 / np.clip(yerr, 1e-12, None) ** 2
    Sxx = np.sum(w * x * x)
    m = np.sum(w * x * y) / Sxx
    resid = y - m * x
    dof = max(len(x) - 1, 1)
    chi2_red = np.sum(w * resid * resid) / dof
    m_err = np.sqrt(chi2_red / Sxx)
    return m, m_err, chi2_red


def fit_with_intercept(x, y, yerr):
    """Ordinary D = m x + b fit (diagnostic: the offset should be ~0)."""
    w = 1.0 / np.clip(yerr, 1e-12, None) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ (A.T @ W @ y)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    cov *= np.sum(w * resid * resid) / dof
    return coef[0], coef[1], np.sqrt(np.diag(cov))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_beads(run):
    """Merge per-bead D (msd.csv) with radius (radius.csv), keep clean singles.

    Resolves a single coherent data directory per run (run root or pipeline/)
    and accepts either label schema, exactly like the plot1 loader. Returns a
    DataFrame with particle, run, r_um, r_err_um, D_um2_s, D_err.
    """
    d = p1.coherent_dir(run)
    msd = pd.read_csv(os.path.join(d, "msd.csv"))
    rad = pd.read_csv(os.path.join(d, "radius.csv"))

    df = msd.merge(rad, on="particle", how="inner")
    df = df[df["r_um"].notna() & df["D_um2_s"].notna() & (df["D_um2_s"] > 0)]

    lp = os.path.join(d, "labels.csv")
    if os.path.exists(lp):
        clean = p1.clean_particle_set(pd.read_csv(lp))
        if clean:
            df = df[df["particle"].isin(clean)]

    # A rough radius uncertainty from frame-to-frame scatter, when a CV column
    # is present (root radius.csv -> r_px_frame_cv; pipeline radius.csv -> R_cv).
    cv_col = next((c for c in ("r_px_frame_cv", "R_cv") if c in df.columns), None)
    n_meas = df["n_meas"] if "n_meas" in df.columns else 1
    if cv_col is not None:
        df["r_err_um"] = (df["r_um"] * df[cv_col].clip(0, 0.5)
                          / np.sqrt(np.clip(n_meas, 1, None)))
    else:
        df["r_err_um"] = 0.0

    if "D_err" not in df.columns:
        df["D_err"] = df["D_um2_s"] * 0.05

    df["run"] = run
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def make_figure(free, hindered, slope, slope_err, kB, kB_err, run,
                T_C, eta_cP, r_star, use_cut):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    # fitted free-diffusion beads
    ax.errorbar(free["inv_r"], free["D_um2_s"],
                xerr=free["inv_r_err"], yerr=free["D_err"],
                fmt="o", color="#1f77b4", ms=5, capsize=2, lw=1,
                ecolor="#9ecae1", label=f"free beads (fit, n={len(free)})")

    # wall-hindered beads (excluded)
    if use_cut and len(hindered):
        ax.errorbar(hindered["inv_r"], hindered["D_um2_s"],
                    xerr=hindered["inv_r_err"], yerr=hindered["D_err"],
                    fmt="s", mfc="none", mec="#888888", ms=5, capsize=2,
                    lw=1, ecolor="#cccccc",
                    label=rf"wall-hindered, $r>r^*$ (n={len(hindered)})")

    # through-origin Stokes-Einstein line
    x_line = np.array([0.0, free["inv_r"].max() * 1.05])
    ax.plot(x_line, slope * x_line, "-", color="#d62728", lw=1.8,
            label=r"$D = (k_BT/6\pi\eta)\,(1/r)$")
    # +/- 1 sigma slope band
    ax.fill_between(x_line, (slope - slope_err) * x_line,
                    (slope + slope_err) * x_line,
                    color="#d62728", alpha=0.12)

    ax.set_xlabel(r"inverse radius  $1/r$  [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"diffusion coefficient  $D$  [$\mu$m$^2$/s]")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Stokes-Einstein: $D \\propto 1/r$  ({run})")

    ratio = kB / K_B_ACCEPTED
    txt = (rf"$k_B = ({kB*1e23:.2f}\pm{kB_err*1e23:.2f})\times10^{{-23}}$ J/K"
           "\n"
           rf"$= {ratio:.2f}\,k_B^{{\rm acc}}$"
           "\n"
           rf"slope $={slope:.4f}\,\mu$m$^3$/s"
           "\n"
           rf"$T={T_C:.1f}\,^\circ$C,  $\eta={eta_cP:.3f}$ cP")
    if use_cut:
        txt += "\n" + rf"$r^*={r_star:.2f}\,\mu$m"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run3")
    ap.add_argument("--T", type=float, default=25.0, help="temperature [C]")
    ap.add_argument("--eta", type=float, default=None,
                    help="viscosity [cP] (default: water at T)")
    ap.add_argument("--delta-rho", type=float, default=50.0,
                    help="|bead-fluid| density mismatch [kg/m^3] for r*")
    ap.add_argument("--r-star", type=float, default=None,
                    help="override the free-diffusion radius cut [um]")
    ap.add_argument("--no-cut", action="store_true",
                    help="fit every clean bead (no wall-hindrance cut)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eta_cP = args.eta if args.eta is not None else water_viscosity_cP(args.T)
    eta_Pa_s = eta_cP * 1e-3
    r_star = args.r_star if args.r_star is not None \
        else sediment_r_star_um(args.T, args.delta_rho)
    use_cut = not args.no_cut

    df = load_beads(args.run)
    df["inv_r"] = 1.0 / df["r_um"]
    df["inv_r_err"] = df["r_err_um"] / df["r_um"] ** 2

    if use_cut:
        free = df[df["r_um"] <= r_star].copy()
        hindered = df[df["r_um"] > r_star].copy()
    else:
        free, hindered = df.copy(), df.iloc[0:0].copy()

    if len(free) < 2:
        raise SystemExit(f"need >=2 free beads to fit, found {len(free)} "
                         f"(r* = {r_star:.2f} um)")

    slope, slope_err, chi2 = fit_through_origin(
        free["inv_r"].values, free["D_um2_s"].values, free["D_err"].values)
    kB = kB_from_slope(slope, args.T, eta_Pa_s)
    kB_err = kB_from_slope(slope_err, args.T, eta_Pa_s)

    # diagnostics: free fit + robust per-bead median
    m2, b2, (m2e, b2e) = fit_with_intercept(
        free["inv_r"].values, free["D_um2_s"].values, free["D_err"].values)
    kb_i = kB_per_bead(free["D_um2_s"].values, free["r_um"].values,
                       args.T, eta_Pa_s)
    kb_median = float(np.median(kb_i))

    print(f"\n{args.run}: T={args.T:.1f}C  eta={eta_cP:.3f}cP  "
          f"r*={r_star:.2f}um  cut={'on' if use_cut else 'off'}")
    print(f"beads: total clean={len(df)}  free(fit)={len(free)}  "
          f"hindered={len(hindered)}")
    print(f"through-origin slope = {slope:.5f} +/- {slope_err:.5f} um^3/s  "
          f"(chi2_red={chi2:.2f})")
    print(f"  -> k_B = {kB:.4e} +/- {kB_err:.2e} J/K  "
          f"({kB/K_B_ACCEPTED:.3f} x accepted)")
    print(f"free fit w/ intercept: slope={m2:.5f}+/-{m2e:.5f}, "
          f"intercept={b2:.4f}+/-{b2e:.4f} um^2/s (should be ~0)")
    print(f"per-bead median k_B = {kb_median:.4e} J/K "
          f"({kb_median/K_B_ACCEPTED:.3f} x accepted)")

    set_style()
    fig = make_figure(free, hindered, slope, slope_err, kB, kB_err,
                      args.run, args.T, eta_cP, r_star, use_cut)

    out = args.out or os.path.join(MEAS, args.run, "figures",
                                   "plot2_D_vs_inv_r.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved -> {out}\n")


if __name__ == "__main__":
    main()
