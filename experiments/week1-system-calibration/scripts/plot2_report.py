"""
plot2_report.py  (week1-system-calibration)
-------------------------------------------
Plot 2 (booklet Part 1): Stokes-Einstein  D = k_B T / (6 pi eta r).

Pools the CURATED single spheres (label = perfect/singlet from labels.csv),
takes each bead's D from msd.csv and physical radius r from radius.csv, plots
D vs 1/r, fits a line THROUGH THE ORIGIN (SE has no intercept), and extracts

        k_B = 6 pi eta (slope) / T          [J/K]

slope is in um^3/s (D in um^2/s vs 1/r in 1/um) -> x1e-18 for m^3/s.
Doublets are shown but EXCLUDED (a rigid pair has r_eff > its main sphere, so it
sits below the single-sphere line); blobs are dropped entirely.

Temperature/viscosity: pass --temp-C (default 25); eta defaults to water at that
T via physics.water_viscosity_cP, or override with --eta-cP. k_B scales as eta/T,
so a measured T tightens the result.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/plot2_report.py run3
    python scripts/plot2_report.py run3 --tag d21m600 --temp-C 24
    python scripts/plot2_report.py run3 --include-doublets   # also fit doublets
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths
import figure_style
import physics


def main():
    ap = argparse.ArgumentParser(description="Plot 2: D vs 1/r -> k_B (Stokes-Einstein).")
    ap.add_argument("run", help="run stem, e.g. run3")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--temp-C", type=float, default=25.0, help="temperature (deg C)")
    ap.add_argument("--eta-cP", type=float, default=None, help="viscosity (cP); default=water at temp")
    ap.add_argument("--include-doublets", action="store_true",
                    help="also include doublets in the fit (default: shown but excluded)")
    args = ap.parse_args()

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    need = ["labels.csv", "radius.csv", "msd.csv"]
    for f in need:
        if not os.path.exists(os.path.join(cdir, f)):
            sys.exit(f"missing {f} in {cdir} (need track->msd_fit->measure_radius->label_beads)")

    lab = pd.read_csv(os.path.join(cdir, "labels.csv"))
    rad = pd.read_csv(os.path.join(cdir, "radius.csv"))
    msd = pd.read_csv(os.path.join(cdir, "msd.csv"))
    df = lab.merge(rad[["particle", "r_um"]], on="particle", how="left", suffixes=("", "_rad"))
    df = df.merge(msd[["particle", "D_um2_s"]], on="particle", how="left")
    df["r"] = df["r_um_rad"].fillna(df.get("r_um"))
    df["invr"] = 1.0 / df["r"]

    singles = df[df["type"].isin(["perfect", "singlet"])].dropna(subset=["r", "D_um2_s"])
    doublets = df[df["type"] == "doublet"].dropna(subset=["r", "D_um2_s"])
    fitset = pd.concat([singles, doublets]) if args.include_doublets else singles

    x = fitset["invr"].values
    y = fitset["D_um2_s"].values
    slope = float(np.sum(x * y) / np.sum(x * x))          # through origin
    r2 = 1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
    resid = y - slope * x
    se = float(np.sqrt(np.sum(resid ** 2) / (len(x) - 1) / np.sum(x * x)))

    eta_cP = args.eta_cP if args.eta_cP is not None else physics.water_viscosity_cP(args.temp_C)
    eta = eta_cP * 1e-3                                    # Pa.s
    T = args.temp_C + 273.15
    kB = 6 * np.pi * eta * (slope * 1e-18) / T
    kB_err = 6 * np.pi * eta * (se * 1e-18) / T
    print(f"[plot2] n_single={len(singles)} n_doublet={len(doublets)} "
          f"(fit on {'singles+doublets' if args.include_doublets else 'singles'})")
    print(f"[plot2] slope={slope:.4f} um^3/s  R2={r2:.2f}  T={args.temp_C}C eta={eta_cP:.2f}cP")
    print(f"[plot2] k_B = {kB:.3e} +/- {kB_err:.1e} J/K  (accepted 1.381e-23; ratio {kB/physics.K_B:.2f})")

    figure_style.set_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(doublets):
        ax.scatter(doublets["invr"], doublets["D_um2_s"], s=30, facecolor="none",
                   edgecolor="gray", zorder=2,
                   label=f"doublets ({'fit' if args.include_doublets else 'excluded'}, n={len(doublets)})")
    ax.scatter(singles["invr"], singles["D_um2_s"], s=55, color="C0", zorder=3,
               label=f"single spheres (n={len(singles)})")
    xs = np.linspace(0, x.max() * 1.05, 50)
    ax.plot(xs, slope * xs, "r-", lw=2, zorder=4,
            label=f"S\u2013E fit (origin): slope={slope:.3f} $\\mu$m$^3$/s, $R^2$={r2:.2f}")
    ax.set_xlabel(r"$1/r$   [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"$D$   [$\mu$m$^2$/s]")
    ax.set_title("Stokes–Einstein:  " + r"$D = \dfrac{k_B T}{6\pi\eta}\,\dfrac{1}{r}$")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    txt = (f"$k_B = {kB:.2e}$ J/K\n(accepted $1.38\\times10^{{-23}}$; ratio {kB/physics.K_B:.2f})\n"
           f"$T={args.temp_C:.0f}\\,^\\circ$C, $\\eta={eta_cP:.2f}$ cP")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)
    name = f"plot2_report_{stem}{'_' + args.tag if args.tag else ''}.png"
    path = figure_style.savefig(name, fig=fig)
    plt.close(fig)
    print(f"[plot2] wrote {path}")


if __name__ == "__main__":
    main()
