"""
plot2_report.py  (week1-system-calibration)
-------------------------------------------
Plot 2 (booklet Part 1): Stokes-Einstein  D = k_B T / (6 pi eta r).

Pools the CURATED single spheres (label = perfect/singlet from labels.csv),
takes each bead's D from msd.csv and physical radius r from radius.csv, plots
D vs 1/r, and extracts k_B.

ESTIMATORS  (all reported so they can be compared)
--------------------------------------------------
Each single sphere is an INDEPENDENT estimate, k_B,i = 6 pi eta r_i D_i / T.
  - per-bead MEDIAN        : robust headline. Immune to the small/large tilt and
                             to outliers (one bad short track can't move it).
  - per-bead WEIGHTED MEAN : inverse-variance weighted by each bead's propagated
                             error (long tracks + round beads count more), after
                             a 3-sigma MAD clip.
  - SIMPLE slope (origin)  : D = m*(1/r). LEGACY. Through-origin least squares
                             weights every bead by (1/r)^2, so the smallest,
                             noisiest beads dominate ~20:1 -> biased HIGH. Kept
                             only for comparison; do NOT report this as k_B.
  - RADIUS-OFFSET (delta)  : D = m/(r - delta). Absorbs a constant outer-edge
                             over-read; tends to OVER-correct because wall-
                             hindered big beads sit below the line and fake a
                             larger delta. See kb_vs_size.py for why.

CURATION (quality gates -- remove bad MEASUREMENTS, not inconvenient physics)
----------------------------------------------------------------------------
  n_frames        >= --min-frames   (track length -> D reliability)
  |intercept_um2| <  --max-intercept(localisation/blur offset stays small)
  circ_resid_frac <  --max-resid    (roundness; non-spheres out)
  inlier_frac     >  --min-inlier   (clean circle fit)
  r_px_frame_cv   <  --max-rcv      (stable apparent size -> no z-motion/defocus)
  --r-lo <= r <= --r-hi             (optional size window; default = full range)
Pass --no-gate to reproduce the old ungated number (the +19% high one).

Doublets are shown but EXCLUDED; blobs dropped entirely.

Temperature/viscosity: pass --temp-C (default 25); eta defaults to water at that
T via physics.water_viscosity_cP, or override with --eta-cP. k_B scales as eta/T.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/plot2_report.py run3
    python scripts/plot2_report.py run3 --temp-C 23.5
    python scripts/plot2_report.py run3 --no-gate          # legacy ungated
    python scripts/plot2_report.py run3 --r-lo 0.9 --r-hi 2.2
"""

import argparse
import operator
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


def origin_slope(x, y):
    """Through-origin LS slope of y vs x, plus R^2. (Legacy estimator.)"""
    slope = float(np.sum(x * y) / np.sum(x * x))
    r2 = 1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
    se = float(np.sqrt(np.sum((y - slope * x) ** 2) / (len(x) - 1) / np.sum(x * x)))
    return slope, r2, se


def robust_kb(kb_i, kb_err, clip=3.0):
    """Per-bead k_B -> (median, median_SE, wmean, wmean_SE, keep_mask).

    median + MAD-based SE is the robust headline; the inverse-variance weighted
    mean uses the propagated per-bead errors after a `clip`-sigma MAD rejection.
    """
    kb_i = np.asarray(kb_i, float)
    med = float(np.median(kb_i))
    mad = float(np.median(np.abs(kb_i - med))) * 1.4826
    se_med = mad / np.sqrt(len(kb_i)) if len(kb_i) else np.nan
    keep = np.abs(kb_i - med) < clip * mad if mad > 0 else np.ones(len(kb_i), bool)
    w = 1.0 / np.clip(np.asarray(kb_err, float)[keep], 1e-30, None) ** 2
    wmean = float(np.sum(w * kb_i[keep]) / np.sum(w))
    # SE from BOTH formal weights and observed scatter; report the larger (honest)
    se_formal = float(np.sqrt(1.0 / np.sum(w)))
    se_scatter = float(np.std(kb_i[keep], ddof=1) / np.sqrt(keep.sum())) if keep.sum() > 1 else np.nan
    se_w = np.nanmax([se_formal, se_scatter])
    return med, se_med, wmean, se_w, keep


def main():
    ap = argparse.ArgumentParser(description="Plot 2: D vs 1/r -> k_B (Stokes-Einstein).")
    ap.add_argument("run", help="run stem, e.g. run3")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--temp-C", type=float, default=25.0, help="temperature (deg C)")
    ap.add_argument("--eta-cP", type=float, default=None, help="viscosity (cP); default=water at temp")
    ap.add_argument("--include-doublets", action="store_true",
                    help="also include doublets in the fit (default: shown but excluded)")
    ap.add_argument("--no-delta", action="store_true", help="skip the radius-offset (delta) fit")
    # ---- curation gates ----
    ap.add_argument("--no-gate", action="store_true", help="disable all quality gates (legacy ungated)")
    ap.add_argument("--min-frames", type=int, default=300, help="drop tracks shorter than this")
    ap.add_argument("--max-intercept", type=float, default=0.25, help="max |MSD intercept| (um^2)")
    ap.add_argument("--max-resid", type=float, default=0.08, help="max circ_resid_frac (roundness)")
    ap.add_argument("--min-inlier", type=float, default=0.70, help="min circle-fit inlier_frac")
    ap.add_argument("--max-rcv", type=float, default=0.15, help="max r_px_frame_cv (focus stability)")
    ap.add_argument("--r-lo", type=float, default=0.0, help="min radius (um); default no cut")
    ap.add_argument("--r-hi", type=float, default=np.inf, help="max radius (um); default no cut")
    args = ap.parse_args()

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    for f in ["radius.csv", "msd.csv"]:
        if not os.path.exists(os.path.join(cdir, f)):
            sys.exit(f"missing {f} in {cdir} (need track->msd_fit->measure_radius)")

    rad = pd.read_csv(os.path.join(cdir, "radius.csv"))
    msd = pd.read_csv(os.path.join(cdir, "msd.csv"))
    rcols = [c for c in ["particle", "r_um", "circ_resid_frac", "inlier_frac", "r_px_frame_cv"] if c in rad]
    mcols = [c for c in ["particle", "D_um2_s", "D_err", "n_frames", "intercept_um2", "size_cv"] if c in msd]
    df = rad[rcols].merge(msd[mcols], on="particle", how="inner")
    # labels.csv is OPTIONAL: a cross-check, not required. When absent, single
    # spheres are identified objectively from the roundness gate below.
    lpath = os.path.join(cdir, "labels.csv")
    if os.path.exists(lpath):
        df = df.merge(pd.read_csv(lpath)[["particle", "type"]], on="particle", how="left")
    df = df.rename(columns={"r_um": "r"})
    df["invr"] = 1.0 / df["r"]

    mpp = _paths.load_scale() or 0.1438
    eta_cP = args.eta_cP if args.eta_cP is not None else physics.water_viscosity_cP(args.temp_C)
    eta = eta_cP * 1e-3                                    # Pa.s
    T = args.temp_C + 273.15
    pref = 6 * np.pi * eta * 1e-18 / T                    # k_B = pref * slope(um^3/s)

    # per-bead k_B and its propagated error (relative errors add in quadrature)
    df["kB_i"] = pref * df["D_um2_s"] * df["r"]
    rel_D = (df["D_err"] / df["D_um2_s"]).abs() if "D_err" in df else 0.0
    rel_r = df["r_px_frame_cv"].fillna(0.0) if "r_px_frame_cv" in df else 0.0
    df["kB_i_err"] = df["kB_i"] * np.sqrt(rel_D ** 2 + rel_r ** 2)

    # ---- curation -------------------------------------------------------------
    def gate(col, op, val):
        return op(df[col], val) if col in df else np.ones(len(df), bool)

    if args.no_gate:
        passes = np.ones(len(df), bool)
    else:
        passes = (gate("n_frames", operator.ge, args.min_frames)
                  & (df["intercept_um2"].abs() < args.max_intercept if "intercept_um2" in df else True)
                  & gate("circ_resid_frac", operator.lt, args.max_resid)
                  & gate("inlier_frac", operator.gt, args.min_inlier)
                  & gate("r_px_frame_cv", operator.lt, args.max_rcv)
                  & (df["r"] >= args.r_lo) & (df["r"] <= args.r_hi))
    df["passes"] = passes

    if "type" in df:                       # manual labels available (cross-check)
        is_single = df["type"].isin(["perfect", "singlet"])
        doublets = df[df["type"] == "doublet"].dropna(subset=["r", "D_um2_s"])
    else:                                  # objective: round + clean circle fit
        is_single = ((df.get("circ_resid_frac", 0) < args.max_resid)
                     & (df.get("inlier_frac", 1) > args.min_inlier))
        doublets = df.iloc[0:0]
    singles = df[is_single & passes].dropna(subset=["r", "D_um2_s"])
    excluded_singles = df[is_single & ~passes].dropna(subset=["r", "D_um2_s"])
    fitset = pd.concat([singles, doublets]) if args.include_doublets else singles
    if len(fitset) < 2:
        sys.exit(f"only {len(fitset)} beads pass the gates -- loosen them or check labels.csv")

    x = fitset["invr"].values
    y = fitset["D_um2_s"].values
    r = fitset["r"].values

    # ---- estimators -----------------------------------------------------------
    med, se_med, wmean, se_w, keep = robust_kb(fitset["kB_i"].values, fitset["kB_i_err"].values)
    slope, r2, se = origin_slope(x, y)
    kB_slope, kB_slope_err = pref * slope, pref * se

    # legacy ungated origin-LS, for the before/after story
    leg = df[is_single].dropna(subset=["r", "D_um2_s"])
    leg_slope, leg_r2, _ = origin_slope(leg["invr"].values, leg["D_um2_s"].values)
    kB_legacy = pref * leg_slope

    print(f"[plot2] {stem}: T={args.temp_C}C eta={eta_cP:.2f}cP mpp={mpp:.5f}  "
          f"gates={'OFF' if args.no_gate else 'ON'}")
    print(f"        n_single(pass)={len(singles)}  n_single(excluded)={len(excluded_singles)}  "
          f"n_doublet={len(doublets)}")
    print(f"  [HEADLINE per-bead median ] k_B={med:.3e} +/- {se_med:.1e}  ratio={med/physics.K_B:.3f}")
    wflag = ("  <- precision-weighted; pulled LOW because the most-precise (longest"
             "-track) beads are the LARGE wall-hindered ones") if wmean < 0.9 * med else ""
    print(f"  [per-bead weighted mean   ] k_B={wmean:.3e} +/- {se_w:.1e}  ratio={wmean/physics.K_B:.3f}"
          f"  ({keep.sum()}/{len(fitset)} after clip){wflag}")
    print(f"  [slope origin (gated)     ] k_B={kB_slope:.3e} +/- {kB_slope_err:.1e}  ratio={kB_slope/physics.K_B:.3f}  R2={r2:.2f}")
    print(f"  [slope origin (LEGACY all)] k_B={kB_legacy:.3e}  ratio={kB_legacy/physics.K_B:.3f}  R2={leg_r2:.2f}  <- the +19% artifact")

    # (2) radius-offset delta fit (on the gated set)
    md = delta = kB_d = None
    if not args.no_delta and len(fitset) >= 3:
        try:
            from scipy.optimize import curve_fit
            model = lambda rr, m, d: m / (rr - d)
            popt, _ = curve_fit(model, r, y, p0=[slope, 0.1],
                                bounds=([0.0, 0.0], [np.inf, 0.95 * r.min()]), maxfev=20000)
            md, delta = float(popt[0]), float(popt[1])
            kB_d = pref * md
            print(f"  [radius-offset delta      ] m={md:.4f}  delta={delta:.3f}um (={delta/mpp:.2f}px)  "
                  f"k_B={kB_d:.3e}  ratio={kB_d/physics.K_B:.3f}")
        except Exception as e:
            print(f"  [radius-offset delta      ] skipped ({e})")

    # ---- plot ----
    figure_style.set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5))
    if len(excluded_singles):
        ax.scatter(excluded_singles["invr"], excluded_singles["D_um2_s"], s=28, facecolor="none",
                   edgecolor="0.6", zorder=2, label=f"excluded by gates (n={len(excluded_singles)})")
    if len(doublets):
        ax.scatter(doublets["invr"], doublets["D_um2_s"], s=28, marker="x",
                   color="gray", zorder=2,
                   label=f"doublets ({'fit' if args.include_doublets else 'excl.'}, n={len(doublets)})")
    ax.scatter(singles["invr"], singles["D_um2_s"], s=55, color="C0", zorder=4,
               label=f"single spheres (n={len(singles)})")
    xs = np.linspace(1e-3, fitset["invr"].max() * 1.05, 200)
    # headline: robust median k_B as a line through origin (slope = k_B/pref)
    ax.plot(xs, (med / pref) * xs, "C0-", lw=2.4, zorder=5,
            label=f"per-bead median: $k_B$={med:.2e} ({med/physics.K_B:.2f}$\\times$)")
    ax.plot(xs, slope * xs, "r--", lw=1.6, zorder=3,
            label=f"slope-fit (gated): $k_B$={kB_slope:.2e} ($R^2$={r2:.2f})")
    if kB_d is not None:
        rr = 1.0 / xs
        valid = rr > delta
        ax.plot(xs[valid], md / (rr[valid] - delta), "g:", lw=1.6, zorder=3,
                label=f"offset $\\delta$={delta/mpp:.1f}px: $k_B$={kB_d:.2e}")
    ax.set_xlabel(r"$1/r$   [$\mu$m$^{-1}$]")
    ax.set_ylabel(r"$D$   [$\mu$m$^2$/s]")
    ax.set_title("Stokes–Einstein:  " + r"$D = \dfrac{k_B T}{6\pi\eta}\,\dfrac{1}{r}$")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    txt = (f"accepted $k_B=1.38\\times10^{{-23}}$\n"
           f"$T={args.temp_C:.1f}\\,^\\circ$C, $\\eta={eta_cP:.2f}$ cP, mpp={mpp:.4f}\n"
           f"gates {'OFF' if args.no_gate else 'ON'}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)
    path = figure_style.savefig("plot2.png", fig=fig, outdir=os.path.join(cdir, "figures"))
    plt.close(fig)
    print(f"[plot2] wrote {path}")


if __name__ == "__main__":
    main()
