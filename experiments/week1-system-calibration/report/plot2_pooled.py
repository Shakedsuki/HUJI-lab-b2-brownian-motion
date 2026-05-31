#!/usr/bin/env python3
"""
plot2_pooled.py
===============
Report figure 2, pooled version -- Stokes-Einstein D vs 1/r combining the clean
beads from several same-temperature runs into a single fit.

Why pool
--------
Each run alone has only a handful of free-diffusion beads (small r), so the
through-origin slope -- and hence k_B -- is poorly constrained. Runs 3, 4 and 6
were all taken at room temperature, so their beads share one T and one eta and
can be pooled. The combined fit has far more leverage at small r and yields one
headline k_B.

        D = (k_B T / 6 pi eta) (1/r)   =>   k_B = slope * 6 pi eta / T

Buoyancy-driven wall-hindered beads (r > r*) sub-diffuse and are drawn in grey,
held out of the fit. Free beads are coloured by run so any run-to-run offset is
visible.

This builds on plot2_D_vs_inv_r.py (physics + fits) and plot1_msd_vs_lag.py
(per-run loaders); it does not reuse the old archive code.

Usage
-----
    python plot2_pooled.py                       # runs 3,4,6 ; T=25 C
    python plot2_pooled.py --runs run3 run4 --T 25
    python plot2_pooled.py --no-cut              # fit every clean bead
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import plot2_D_vs_inv_r as p2     # physics, fits, loader, style


RUN_COLORS = {                    # stable per-run colours for the scatter
    "run3": "#1f77b4", "run4": "#2ca02c", "run6": "#9467bd",
    "run5": "#ff7f0e", "run7": "#17becf", "run8": "#bcbd22",
    "run9": "#e377c2", "run10": "#8c564b",
}


def build_figure(runs, args, eta_cP, eta_Pa_s, r_star, pooled, out):
    """Load, clean, fit and render one D-vs-1/r figure for the given run list.

    `pooled` only affects the title/labelling; the analysis is identical whether
    `runs` holds one run (a single-run figure) or several (a pooled figure).
    """
    # ---- pool the runs ----
    df = pd.concat([p2.load_beads(r) for r in runs], ignore_index=True)
    df["inv_r"] = 1.0 / df["r_um"]
    df["inv_r_err"] = df["r_err_um"] / df["r_um"] ** 2

    # Drop gross mislinks: under Stokes-Einstein D*r is ~constant, so a robust
    # MAD cut on D*r removes the handful of beads whose tracked D is physically
    # impossible (bad links / merges) without touching the genuine scatter.
    k = (df["D_um2_s"] * df["r_um"]).values
    med = np.median(k)
    mad = np.median(np.abs(k - med))
    n_raw = len(df)
    if mad > 0:
        df = df[np.abs(k - med) <= args.mad_k * mad].copy()
    n_drop = n_raw - len(df)

    # By default ALL clean beads are part of the data and the fit; r* is kept
    # only as a reference line. --exclude-hindered restores the free-only fit.
    if args.exclude_hindered:
        free = df[df["r_um"] <= r_star].copy()
        hindered = df[df["r_um"] > r_star].copy()
    else:
        free, hindered = df.copy(), df.iloc[0:0].copy()

    if len(free) < 2:
        raise SystemExit(f"need >=2 beads to fit, found {len(free)}")

    # ---- pooled slope estimators ----
    # The formal D_err (from correlated MSD lags, varying with track length)
    # span ~2 orders of magnitude, so inverse-variance weighting is unreliable.
    # The robust headline is the per-bead median slope (= median of D_i*r_i, a
    # Theil-Sen through-origin slope) -- the same estimator the lab pipeline
    # uses, insensitive to the bad weights and to mislinked outliers. The
    # ordinary unweighted LS fit and the weighted fit are kept as diagnostics.
    x, y, ye = (free["inv_r"].values, free["D_um2_s"].values,
                free["D_err"].values)
    n = len(free)

    per_bead_slope = y / x                       # D_i * r_i, in um^3/s
    slope = float(np.median(per_bead_slope))     # robust headline slope
    mad = np.median(np.abs(per_bead_slope - slope))
    sigma = 1.4826 * mad                          # robust sigma
    slope_err = 1.2533 * sigma / np.sqrt(n)       # SE of the median
    kB = p2.kB_from_slope(slope, args.T, eta_Pa_s)
    kB_err = p2.kB_from_slope(slope_err, args.T, eta_Pa_s)

    slope_ols = float(np.sum(x * y) / np.sum(x * x))      # unweighted thru-0
    slope_wls, _, chi2 = p2.fit_through_origin(x, y, ye)  # inverse-variance

    # ---- report (incl. per-run robust slopes for comparison) ----
    def kbx(s):
        return p2.kB_from_slope(s, args.T, eta_Pa_s) / p2.K_B_ACCEPTED

    print(f"\n{'pooled ' if pooled else ''}{runs}: T={args.T:.1f}C  "
          f"eta={eta_cP:.3f}cP  r*={r_star:.2f}um  "
          f"hindered={'excluded' if args.exclude_hindered else 'kept'}")
    print(f"beads: kept={len(df)} (dropped {n_drop} D*r mislink outliers)  "
          f"fit n={n}  hindered={len(hindered)}")
    for r in runs:
        g = free[free["run"] == r]
        if len(g):
            sr = float(np.median(g["D_um2_s"].values * g["r_um"].values))
            print(f"  {r}: n_free={len(g)}  median-slope={sr:.4f}  "
                  f"k_B={kbx(sr):.2f}x")
    print(f"median slope = {slope:.5f} +/- {slope_err:.5f} um^3/s")
    print(f"  -> k_B = {kB:.4e} +/- {kB_err:.2e} J/K  "
          f"({kB/p2.K_B_ACCEPTED:.3f} x accepted)  [HEADLINE]")
    print(f"diagnostics: unweighted-LS k_B={kbx(slope_ols):.3f}x  "
          f"weighted-LS k_B={kbx(slope_wls):.3f}x (chi2_red={chi2:.0f})")

    # ---- figure ----
    p2.set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # run-coloured points, no per-point error bars (a single representative bar
    # is drawn in the corner instead, so the cloud reads cleanly)
    for r in runs:
        g = free[free["run"] == r]
        if not len(g):
            continue
        ax.plot(g["inv_r"], g["D_um2_s"], "o", ms=4, alpha=0.7,
                color=RUN_COLORS.get(r, "#444444"), mec="none",
                label=f"{r} (n={len(g)})")

    # headline: robust (per-bead median) through-origin line + SE band
    x_line = np.array([0.0, free["inv_r"].max() * 1.05])
    ax.plot(x_line, slope * x_line, "-", color="#d62728", lw=2.0, zorder=5,
            label=r"robust fit  $D=(k_BT/6\pi\eta)\,(1/r)$")
    ax.fill_between(x_line, (slope - slope_err) * x_line,
                    (slope + slope_err) * x_line,
                    color="#d62728", alpha=0.13, zorder=1)

    ax.set_xlabel(r"inverse radius  $1/r$  [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"diffusion coefficient  $D$  [$\mu$m$^2$/s]")
    ax.set_xlim(left=0.0, right=free["inv_r"].max() * 1.08)
    ax.set_ylim(bottom=0.0, top=free["D_um2_s"].max() * 1.12)
    title = ("Stokes-Einstein (pooled room-temperature runs)" if pooled
             else f"Stokes-Einstein ({runs[0]})")
    ax.set_title(title)

    # one representative error bar in the empty centre-left region
    xr, yr = ax.get_xlim(), ax.get_ylim()
    ex = float(np.median(free["inv_r_err"]))
    ey = float(np.median(free["D_err"]))
    ex0, ey0 = xr[1] * 0.24, yr[1] * 0.55
    ax.errorbar([ex0], [ey0], xerr=ex, yerr=ey, fmt="o", ms=4,
                color="0.35", ecolor="0.35", capsize=3, lw=1.1)
    ax.text(ex0 + ex * 1.3, ey0, " typical\n uncertainty", color="0.35",
            fontsize=8, ha="left", va="center")

    ratio = kB / p2.K_B_ACCEPTED
    txt = (rf"$k_B = ({kB*1e23:.2f}\pm{kB_err*1e23:.2f})\times10^{{-23}}$ J/K"
           "\n"
           rf"$= {ratio:.2f}\,k_B^{{\rm accepted}}$   ($n={len(free)}$ beads)"
           "\n"
           rf"$T={args.T:.1f}\,^\circ$C,  $\eta={eta_cP:.3f}$ cP"
           "\n"
           rf"(LS slope check: ${kbx(slope_ols):.2f}\,k_B^{{\rm acc}}$)")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["run3", "run4", "run6"])
    ap.add_argument("--split", action="store_true",
                    help="emit one figure per run instead of a single pooled "
                         "figure (same clean design for each)")
    ap.add_argument("--T", type=float, default=25.0, help="temperature [C]")
    ap.add_argument("--eta", type=float, default=None,
                    help="viscosity [cP] (default: water at T)")
    ap.add_argument("--delta-rho", type=float, default=50.0,
                    help="|bead-fluid| density mismatch [kg/m^3] for r*")
    ap.add_argument("--r-star", type=float, default=None,
                    help="override the sedimentation-scale reference r* [um]")
    ap.add_argument("--exclude-hindered", action="store_true",
                    help="drop wall-hindered beads (r > r*) from the fit; by "
                         "default every clean bead is kept as data")
    ap.add_argument("--mad-k", type=float, default=3.5,
                    help="robust D*r outlier cut in MAD units (mislink removal)")
    ap.add_argument("--out", default=None,
                    help="output path (pooled mode only; ignored with --split)")
    args = ap.parse_args()

    eta_cP = args.eta if args.eta is not None else p2.water_viscosity_cP(args.T)
    eta_Pa_s = eta_cP * 1e-3
    r_star = (args.r_star if args.r_star is not None
              else p2.sediment_r_star_um(args.T, args.delta_rho))

    figdir = os.path.join(p2.ROOT, "figures")
    if args.split:
        for run in args.runs:
            out = os.path.join(figdir, f"plot2_{run}.png")
            build_figure([run], args, eta_cP, eta_Pa_s, r_star,
                         pooled=False, out=out)
    else:
        tag = "-".join(r.replace("run", "") for r in args.runs)
        out = args.out or os.path.join(figdir, f"plot2_pooled_runs{tag}.png")
        build_figure(args.runs, args, eta_cP, eta_Pa_s, r_star,
                     pooled=True, out=out)


if __name__ == "__main__":
    main()
