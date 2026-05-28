"""
plot2_report.py  (week1-system-calibration)
-------------------------------------------
Plot 2 (booklet Part 1): Stokes-Einstein  D = k_B T / (6 pi eta r).

Pools the CURATED single spheres (label = perfect/singlet from labels.csv),
takes each bead's D from msd.csv and physical radius r from radius.csv, plots
D vs 1/r, and extracts k_B = 6 pi eta (slope) / T   [J/K].
slope is in um^3/s (D in um^2/s vs 1/r in 1/um) -> x1e-18 for m^3/s.

Two fits:
  (1) SIMPLE, through the origin:  D = m * (1/r).
  (2) RADIUS-OFFSET (Lever 2):     D = m / (r - delta),  delta >= 0 a CONSTANT
      length (um). Our outer-edge radius over-reads the true sphere edge because
      diffraction spreads the dark ring ~a fixed distance beyond the physical
      edge; that offset is additive in r, so it biases SMALL beads most and
      TILTS the line (a slope/k_B bias, not just a scale). Fitting delta absorbs
      that systematic. Diagnostic: if delta ~ 1-2 px and k_B drops toward
      1.38e-23, the radius over-read was the culprit; if delta -> 0 and k_B
      stays high, the bias is elsewhere (temperature or D).
  k_B is reported from BOTH so they can be compared.

Doublets are shown but EXCLUDED (a rigid pair has r_eff > its main sphere -> sits
below the single-sphere line); blobs are dropped entirely.

Temperature/viscosity: pass --temp-C (default 25); eta defaults to water at that
T via physics.water_viscosity_cP, or override with --eta-cP. k_B scales as eta/T.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/plot2_report.py run3
    python scripts/plot2_report.py run3 --temp-C 23.5
    python scripts/plot2_report.py run3 --no-delta            # simple fit only
    python scripts/plot2_report.py run3 --include-doublets
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
    ap.add_argument("--no-delta", action="store_true", help="skip the radius-offset (delta) fit")
    args = ap.parse_args()

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    for f in ["labels.csv", "radius.csv", "msd.csv"]:
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
    r = fitset["r"].values
    mpp = _paths.load_scale() or 0.1438

    eta_cP = args.eta_cP if args.eta_cP is not None else physics.water_viscosity_cP(args.temp_C)
    eta = eta_cP * 1e-3                                    # Pa.s
    T = args.temp_C + 273.15
    pref = 6 * np.pi * eta * 1e-18 / T                    # k_B = pref * slope(um^3/s)

    # (1) simple through-origin
    slope = float(np.sum(x * y) / np.sum(x * x))
    r2 = 1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
    se = float(np.sqrt(np.sum((y - slope * x) ** 2) / (len(x) - 1) / np.sum(x * x)))
    kB = pref * slope
    kB_err = pref * se

    print(f"[plot2] n_single={len(singles)} n_doublet={len(doublets)} "
          f"T={args.temp_C}C eta={eta_cP:.2f}cP mpp={mpp:.5f}")
    print(f"[simple] slope={slope:.4f} um^3/s  R2={r2:.2f}  "
          f"k_B={kB:.3e} +/- {kB_err:.1e}  ratio={kB/physics.K_B:.2f}")

    # (2) radius-offset delta fit
    md = delta = kB_d = None
    if not args.no_delta:
        try:
            from scipy.optimize import curve_fit
            model = lambda rr, m, d: m / (rr - d)
            p0 = [slope, 0.1]
            bounds = ([0.0, 0.0], [np.inf, 0.95 * r.min()])
            popt, _ = curve_fit(model, r, y, p0=p0, bounds=bounds, maxfev=20000)
            md, delta = float(popt[0]), float(popt[1])
            yhat = model(r, md, delta)
            r2_d = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
            kB_d = pref * md
            print(f"[delta ] m={md:.4f} um^3/s  delta={delta:.3f} um (={delta/mpp:.2f} px)  "
                  f"R2={r2_d:.2f}  k_B={kB_d:.3e}  ratio={kB_d/physics.K_B:.2f}")
        except Exception as e:
            print(f"[delta ] skipped ({e})")

    # ---- plot ----
    figure_style.set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5))
    if len(doublets):
        ax.scatter(doublets["invr"], doublets["D_um2_s"], s=30, facecolor="none",
                   edgecolor="gray", zorder=2,
                   label=f"doublets ({'fit' if args.include_doublets else 'excluded'}, n={len(doublets)})")
    ax.scatter(singles["invr"], singles["D_um2_s"], s=55, color="C0", zorder=3,
               label=f"single spheres (n={len(singles)})")
    xs = np.linspace(1e-3, x.max() * 1.05, 200)
    ax.plot(xs, slope * xs, "r-", lw=2, zorder=4,
            label=f"simple (origin): $R^2$={r2:.2f}, $k_B$={kB:.2e}")
    if kB_d is not None:
        rr = 1.0 / xs
        valid = rr > delta
        ax.plot(xs[valid], md / (rr[valid] - delta), "g--", lw=2, zorder=5,
                label=f"offset $\\delta$={delta/mpp:.1f}px: $k_B$={kB_d:.2e}")
    ax.set_xlabel(r"$1/r$   [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"$D$   [$\mu$m$^2$/s]")
    ax.set_title("Stokes\u2013Einstein:  " + r"$D = \dfrac{k_B T}{6\pi\eta}\,\dfrac{1}{r}$")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    txt = (f"accepted $k_B=1.38\\times10^{{-23}}$\n"
           f"$T={args.temp_C:.1f}\\,^\\circ$C, $\\eta={eta_cP:.2f}$ cP, mpp={mpp:.4f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)
    name = f"plot2_report_{stem}{'_' + args.tag if args.tag else ''}.png"
    path = figure_style.savefig(name, fig=fig)
    plt.close(fig)
    print(f"[plot2] wrote {path}")


if __name__ == "__main__":
    main()
