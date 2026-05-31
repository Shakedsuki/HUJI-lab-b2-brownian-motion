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


def analyze(runs, args, eta_cP, eta_Pa_s, r_star, pooled=True, verbose=True):
    """Load, clean and fit the D-vs-1/r data for `runs`; return a result dict.

    Same procedure whether `runs` holds one run or several.
    """
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
    s_mad = np.median(np.abs(per_bead_slope - slope))
    slope_err = 1.2533 * (1.4826 * s_mad) / np.sqrt(n)    # SE of the median
    kB = p2.kB_from_slope(slope, args.T, eta_Pa_s)
    kB_err = p2.kB_from_slope(slope_err, args.T, eta_Pa_s)

    slope_ols = float(np.sum(x * y) / np.sum(x * x))      # unweighted thru-0
    slope_wls, _, chi2 = p2.fit_through_origin(x, y, ye)  # inverse-variance

    def kbx(s):
        return p2.kB_from_slope(s, args.T, eta_Pa_s) / p2.K_B_ACCEPTED

    res = dict(runs=runs, free=free, n=n, n_drop=n_drop,
               n_hindered=len(hindered), slope=slope, slope_err=slope_err,
               kB=kB, kB_err=kB_err, ratio=kB / p2.K_B_ACCEPTED,
               ratio_ols=kbx(slope_ols), eta_cP=eta_cP, pooled=pooled)

    if verbose:
        print(f"\n{'pooled ' if pooled else ''}{runs}: T={args.T:.1f}C  "
              f"eta={eta_cP:.3f}cP  r*={r_star:.2f}um  "
              f"hindered={'excluded' if args.exclude_hindered else 'kept'}")
        print(f"beads: kept={len(df)} (dropped {n_drop} D*r mislinks)  "
              f"fit n={n}  hindered={len(hindered)}")
        for r in runs:
            g = free[free["run"] == r]
            if len(g):
                sr = float(np.median(g["D_um2_s"].values * g["r_um"].values))
                print(f"  {r}: n={len(g)}  k_B={kbx(sr):.2f}x")
        print(f"  -> k_B = {kB:.4e} +/- {kB_err:.2e} J/K "
              f"({res['ratio']:.3f}x)  LS={res['ratio_ols']:.3f}x "
              f"WLS={kbx(slope_wls):.3f}x (chi2={chi2:.0f})")
    return res


def draw_panel(ax, res, args, *, full=True, xmax=None, ymax=None,
               title=None, ylabel=True):
    """Render one D-vs-1/r panel from an analyze() result onto `ax`.

    full=True  -> standalone figure (representative error bar, full stats box,
                  legend). full=False -> compact grid panel (k_B label only).
    """
    free, slope, slope_err = res["free"], res["slope"], res["slope_err"]

    for r in res["runs"]:
        g = free[free["run"] == r]
        if len(g):
            ax.plot(g["inv_r"], g["D_um2_s"], "o", ms=4, alpha=0.7,
                    color=RUN_COLORS.get(r, "#444444"), mec="none",
                    label=f"{r} (n={len(g)})")

    xhi = xmax if xmax is not None else free["inv_r"].max() * 1.08
    yhi = ymax if ymax is not None else free["D_um2_s"].max() * 1.12
    ax.set_xlim(left=0.0, right=xhi)
    ax.set_ylim(bottom=0.0, top=yhi)

    x_line = np.array([0.0, xhi])
    ax.plot(x_line, slope * x_line, "-", color="#d62728", lw=2.0, zorder=5,
            label=r"robust fit  $D=(k_BT/6\pi\eta)\,(1/r)$")
    ax.fill_between(x_line, (slope - slope_err) * x_line,
                    (slope + slope_err) * x_line,
                    color="#d62728", alpha=0.13, zorder=1)

    ax.set_xlabel(r"inverse radius  $1/r$  [$\mu$m$^{-1}$]")
    if ylabel:
        ax.set_ylabel(r"diffusion coefficient  $D$  [$\mu$m$^2$/s]")
    ax.set_title(title if title is not None else
                 (f"Stokes-Einstein ({res['runs'][0]})"))

    if full:
        ex = float(np.median(free["inv_r_err"]))
        ey = float(np.median(free["D_err"]))
        ex0, ey0 = xhi * 0.24, yhi * 0.55
        ax.errorbar([ex0], [ey0], xerr=ex, yerr=ey, fmt="o", ms=4,
                    color="0.35", ecolor="0.35", capsize=3, lw=1.1)
        ax.text(ex0 + ex * 1.3, ey0, " typical\n uncertainty", color="0.35",
                fontsize=8, ha="left", va="center")
        txt = (rf"$k_B = ({res['kB']*1e23:.2f}\pm{res['kB_err']*1e23:.2f})"
               rf"\times10^{{-23}}$ J/K"
               "\n"
               rf"$= {res['ratio']:.2f}\,k_B^{{\rm accepted}}$   "
               rf"($n={res['n']}$ beads)"
               "\n"
               rf"$T={args.T:.1f}\,^\circ$C,  $\eta={res['eta_cP']:.3f}$ cP"
               "\n"
               rf"(LS slope check: ${res['ratio_ols']:.2f}\,k_B^{{\rm acc}}$)")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))
        ax.legend(loc="lower right", fontsize=9)
    else:
        ax.text(0.05, 0.95,
                rf"$k_B={res['ratio']:.2f}\,k_B^{{\rm acc}}$" "\n"
                rf"$n={res['n']}$", transform=ax.transAxes, va="top",
                ha="left", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


def build_figure(runs, args, eta_cP, eta_Pa_s, r_star, pooled, out):
    """Single standalone D-vs-1/r figure for `runs` (pooled or one run)."""
    res = analyze(runs, args, eta_cP, eta_Pa_s, r_star, pooled=pooled)
    p2.set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    title = ("Stokes-Einstein (pooled room-temperature runs)" if pooled
             else f"Stokes-Einstein ({runs[0]})")
    draw_panel(ax, res, args, full=True, title=title)
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
