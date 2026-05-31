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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["run3", "run4", "run6"])
    ap.add_argument("--T", type=float, default=25.0, help="temperature [C]")
    ap.add_argument("--eta", type=float, default=None,
                    help="viscosity [cP] (default: water at T)")
    ap.add_argument("--delta-rho", type=float, default=50.0,
                    help="|bead-fluid| density mismatch [kg/m^3] for the r* "
                         "reference line")
    ap.add_argument("--r-star", type=float, default=None,
                    help="override the sedimentation-scale reference r* [um]")
    ap.add_argument("--exclude-hindered", action="store_true",
                    help="drop wall-hindered beads (r > r*) from the fit; by "
                         "default every clean bead is kept as data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eta_cP = args.eta if args.eta is not None else p2.water_viscosity_cP(args.T)
    eta_Pa_s = eta_cP * 1e-3
    r_star = (args.r_star if args.r_star is not None
              else p2.sediment_r_star_um(args.T, args.delta_rho))

    # ---- pool the runs ----
    df = pd.concat([p2.load_beads(r) for r in args.runs], ignore_index=True)
    df["inv_r"] = 1.0 / df["r_um"]
    df["inv_r_err"] = df["r_err_um"] / df["r_um"] ** 2

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

    print(f"\npooled {args.runs}: T={args.T:.1f}C  eta={eta_cP:.3f}cP  "
          f"r*={r_star:.2f}um  "
          f"hindered={'excluded' if args.exclude_hindered else 'kept'}")
    print(f"beads: total clean={len(df)}  free(fit)={n}  "
          f"hindered={len(hindered)}")
    for r in args.runs:
        g = free[free["run"] == r]
        if len(g):
            sr = float(np.median(g["D_um2_s"].values * g["r_um"].values))
            print(f"  {r}: n_free={len(g)}  median-slope={sr:.4f}  "
                  f"k_B={kbx(sr):.2f}x")
    print(f"POOLED median slope = {slope:.5f} +/- {slope_err:.5f} um^3/s")
    print(f"  -> k_B = {kB:.4e} +/- {kB_err:.2e} J/K  "
          f"({kB/p2.K_B_ACCEPTED:.3f} x accepted)  [HEADLINE]")
    print(f"diagnostics: unweighted-LS k_B={kbx(slope_ols):.3f}x  "
          f"weighted-LS k_B={kbx(slope_wls):.3f}x (chi2_red={chi2:.0f})")

    # ---- figure ----
    p2.set_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    for r in args.runs:
        g = free[free["run"] == r]
        if not len(g):
            continue
        ax.errorbar(g["inv_r"], g["D_um2_s"],
                    xerr=g["inv_r_err"], yerr=g["D_err"],
                    fmt="o", ms=5, capsize=2, lw=1,
                    color=RUN_COLORS.get(r, "#444444"),
                    ecolor=RUN_COLORS.get(r, "#444444"), alpha=0.9,
                    label=f"{r} (n={len(g)})")

    if args.exclude_hindered and len(hindered):
        ax.errorbar(hindered["inv_r"], hindered["D_um2_s"],
                    xerr=hindered["inv_r_err"], yerr=hindered["D_err"],
                    fmt="s", mfc="none", mec="#999999", ms=4, capsize=2,
                    lw=0.8, ecolor="#dddddd",
                    label=rf"wall-hindered $r>r^*$ (excl., n={len(hindered)})")

    x_line = np.array([0.0, free["inv_r"].max() * 1.05])
    # headline: robust (per-bead median) through-origin line + SE band
    ax.plot(x_line, slope * x_line, "-", color="#d62728", lw=1.9,
            label=r"robust fit $D=(k_BT/6\pi\eta)(1/r)$")
    ax.fill_between(x_line, (slope - slope_err) * x_line,
                    (slope + slope_err) * x_line,
                    color="#d62728", alpha=0.12)
    # diagnostic: ordinary unweighted least-squares through origin
    ax.plot(x_line, slope_ols * x_line, "--", color="#555555", lw=1.2,
            label=f"unweighted LS ({kbx(slope_ols):.2f}$\\,k_B^{{\\rm acc}}$)")
    # reference: sedimentation scale r* (1/r*), where wall hindrance sets in
    if 0 < r_star:
        ax.axvline(1.0 / r_star, color="0.6", ls=":", lw=1.1)
        ax.text(1.0 / r_star, ax.get_ylim()[1] * 0.02,
                rf"  $1/r^*$ ($r^*={r_star:.2f}\,\mu$m)", color="0.4",
                fontsize=8, rotation=90, va="bottom", ha="left")

    ax.set_xlabel(r"inverse radius  $1/r$  [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"diffusion coefficient  $D$  [$\mu$m$^2$/s]")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("Stokes-Einstein, pooled room-temperature runs "
                 f"({', '.join(args.runs)})")

    ratio = kB / p2.K_B_ACCEPTED
    txt = (rf"$k_B = ({kB*1e23:.2f}\pm{kB_err*1e23:.2f})\times10^{{-23}}$ J/K"
           "\n"
           rf"$= {ratio:.2f}\,k_B^{{\rm acc}}$  (per-bead median, $n={len(free)}$)"
           "\n"
           rf"robust slope $={slope:.4f}\,\mu$m$^3$/s"
           "\n"
           rf"$T={args.T:.1f}\,^\circ$C, $\eta={eta_cP:.3f}$ cP")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    tag = "-".join(r.replace("run", "") for r in args.runs)
    out = args.out or os.path.join(p2.ROOT, "figures",
                                   f"plot2_pooled_runs{tag}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved -> {out}\n")


if __name__ == "__main__":
    main()
