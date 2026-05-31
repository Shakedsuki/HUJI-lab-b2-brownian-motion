#!/usr/bin/env python3
"""
plot1_msd_vs_lag.py
===================
Report figure 1 -- Mean squared displacement vs lag time.

Goal
----
Show that the microspheres undergo *normal* diffusion: the time-averaged MSD
grows linearly with lag time,

        <r^2>(tau) = 4 D tau + c          (2D projection)

so a linear fit gives the diffusion coefficient D of each bead, and on a
log-log plot every curve has slope 1. Three beads of different radii are
overlaid; the smallest bead (largest D) has the steepest slope, foreshadowing
the Stokes-Einstein result D = k_B T / (6 pi eta r) that plot 2 nails down.

What it does
------------
1. Reads one run's particle trajectories (pixels, drift-subtracted), the
   per-bead radii, and the human "perfect/singlet/..." labels.
2. Picks three clean single beads spanning the radius range (small / mid /
   large) -- or uses the beads passed on the command line.
3. Computes each bead's gap-aware, time-averaged 2D MSD(tau) straight from the
   trajectory, converts to physical units (um^2 vs s), and fits 4 D tau + c
   over the short-lag linear window.
4. Draws a two-panel figure: (left) MSD vs tau with the linear fits, (right)
   the same data log-log with a slope-1 guide line.

This script is self-contained: it does not import the pipeline modules, and it
deliberately does not reuse the older archive/scripts plotting code.

Usage
-----
    python plot1_msd_vs_lag.py                  # auto everything (run3)
    python plot1_msd_vs_lag.py --run run4
    python plot1_msd_vs_lag.py --beads 92 1 12  # explicit small/mid/large
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


# --------------------------------------------------------------------------- #
# Paths / configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # week1-system-calibration/
MEAS = os.path.join(ROOT, "measurements")
SCALE_JSON = os.path.join(ROOT, "calibration", "scale.json")
VIDEOS_META = os.path.join(ROOT, "videos_meta.json")

# Run-name -> video-file key inside videos_meta.json (only the odd ones differ).
VIDEO_KEY = {
    "run7": "run7_aimed_for_15C.avi",
    "run8": "run8_aimed_for_15C_.avi",
    "run9": "run9_aimed_for_5C.avi",
    "run10": "run10_aimed_for_5C.avi",
}

CLEAN_LABELS = {"perfect", "singlet"}   # bead classes we trust as single spheres


# --------------------------------------------------------------------------- #
# Calibration helpers
# --------------------------------------------------------------------------- #
def load_um_per_px() -> float:
    with open(SCALE_JSON) as f:
        return float(json.load(f)["um_per_px"])


def load_fps(run: str) -> float:
    key = VIDEO_KEY.get(run, f"{run}.avi")
    with open(VIDEOS_META) as f:
        meta = json.load(f)
    return float(meta[key]["fps_avg"])


# --------------------------------------------------------------------------- #
# MSD core
# --------------------------------------------------------------------------- #
def time_averaged_msd(frames, x, y, max_lag):
    """Gap-aware time-averaged 2D MSD of one track.

    The track is placed on a dense frame grid (missing frames -> NaN) so that a
    lag of `k` frames is always `k` real frames of time, even across dropouts.
    For each lag we average (dx^2 + dy^2) over every frame pair that is present
    at both ends.

    Returns
    -------
    lags   : ndarray, lag in frames (1..max_lag)
    msd    : ndarray, mean squared displacement in px^2 (NaN where no pairs)
    npairs : ndarray, number of contributing frame pairs (fit weight)
    """
    frames = np.asarray(frames, dtype=int)
    f0, f1 = frames.min(), frames.max()
    n = f1 - f0 + 1

    gx = np.full(n, np.nan)
    gy = np.full(n, np.nan)
    gx[frames - f0] = x
    gy[frames - f0] = y

    max_lag = int(min(max_lag, n - 1))
    lags = np.arange(1, max_lag + 1)
    msd = np.full(max_lag, np.nan)
    npairs = np.zeros(max_lag, dtype=int)

    for i, k in enumerate(lags):
        dx = gx[k:] - gx[:-k]
        dy = gy[k:] - gy[:-k]
        sq = dx * dx + dy * dy
        good = ~np.isnan(sq)
        m = int(good.sum())
        npairs[i] = m
        if m:
            msd[i] = sq[good].mean()
    return lags, msd, npairs


def fit_linear_msd(tau, msd, weights, t_max):
    """Fit <r^2> = 4 D tau + c over lags with tau <= t_max.

    Weighted least squares (weights ~ number of independent pairs). Returns D,
    its 1-sigma error, the intercept c, and the fit mask used.
    """
    mask = (tau <= t_max) & np.isfinite(msd) & (weights > 0)
    t = tau[mask]
    m = msd[mask]
    w = weights[mask].astype(float)

    # Weighted linear regression m = a*t + c  ->  D = a/4.
    W = np.diag(w)
    A = np.vstack([t, np.ones_like(t)]).T
    AtW = A.T @ W
    cov = np.linalg.inv(AtW @ A)
    coef = cov @ (AtW @ m)
    a, c = coef

    # The weights are only relative (~ pair counts), so rescale the parameter
    # covariance by the weighted residual variance s^2 = chi2/dof -- the usual
    # "scaled" fit covariance (np.polyfit/curve_fit convention). The MSD points
    # are correlated across lags, so this is still an optimistic lower bound on
    # the true uncertainty, but it matches the pipeline's reported D_err.
    resid = m - A @ coef
    dof = max(len(t) - 2, 1)
    s2 = float((w * resid * resid).sum() / dof)
    cov *= s2
    a_err = np.sqrt(cov[0, 0])

    return a / 4.0, a_err / 4.0, c, mask


# --------------------------------------------------------------------------- #
# Bead selection
# --------------------------------------------------------------------------- #
def coherent_dir(run: str):
    """Directory whose trajectory/radius/labels share one particle-ID scheme.

    run3 keeps the curated set at the run root; the other runs keep theirs in
    pipeline/ (the root copies, where present, use a different/older linking).
    Prefer the root, fall back to pipeline/. Labels are optional.
    """
    base = os.path.join(MEAS, run)
    cands = (base, os.path.join(base, "pipeline"))
    for need in (("trajectory.csv", "radius.csv", "labels.csv"),
                 ("trajectory.csv", "radius.csv")):
        for d in cands:
            if all(os.path.exists(os.path.join(d, f)) for f in need):
                return d
    raise SystemExit(f"{run}: no directory with trajectory.csv + radius.csv")


def load_inputs(run: str):
    d = coherent_dir(run)
    traj = pd.read_csv(os.path.join(d, "trajectory.csv"))
    radius = pd.read_csv(os.path.join(d, "radius.csv"))
    lp = os.path.join(d, "labels.csv")
    labels = pd.read_csv(lp) if os.path.exists(lp) else None
    mp = os.path.join(d, "msd.csv")
    msd = pd.read_csv(mp) if os.path.exists(mp) else None
    return traj, radius, labels, msd


def clean_particle_set(labels):
    """Particle IDs flagged as clean single spheres, across label schemas.

    run3-style labels carry a `type` column (perfect/singlet/...); the v2
    pipeline labels carry `keep` (1) / `proposed` (single) instead.
    """
    if labels is None:
        return None
    cols = set(labels.columns)
    if "type" in cols:
        return set(labels.loc[labels["type"].isin(CLEAN_LABELS), "particle"])
    if "keep" in cols:
        return set(labels.loc[labels["keep"] == 1, "particle"])
    if "proposed" in cols:
        return set(labels.loc[labels["proposed"] == "single", "particle"])
    return None


BEADS_JSON = os.path.join(HERE, "fig1_beads.json")


def chosen_beads(run, traj, radius, labels, msd, min_frames):
    """Beads for a run's Figure-1 panel: a manual override if listed in
    report/fig1_beads.json (visually confirmed single spheres), else the
    automatic pick. Returns (particle, r_um) sorted small -> large.

    The automatic roundness gates use whatever shape columns a run provides,
    but those metrics are not comparable run-to-run and cannot always tell a
    doublet from a single -- hence the human-checked override file.
    """
    if os.path.exists(BEADS_JSON):
        with open(BEADS_JSON) as f:
            override = json.load(f)
        if run in override:
            beads = [(int(p), radius_lookup(radius, int(p)))
                     for p in override[run]]
            beads.sort(key=lambda b: (np.inf if np.isnan(b[1]) else b[1]))
            return beads
    return pick_three_beads(traj, radius, labels, msd, min_frames)


def pick_three_beads(traj, radius, labels, msd, min_frames):
    """Choose three clean single beads spanning the radius range.

    Restrict to long, well-measured, human-confirmed single spheres, drop beads
    whose D is grossly inconsistent with Stokes-Einstein (D*r far from the
    sample median -- usually a mislink or a wall-stuck bead), then take the
    small / median / large radius among the survivors. Returns a list of
    (particle, r_um) sorted small -> large.
    """
    counts = traj.groupby("particle").size().rename("n_frames")
    cand = radius.merge(counts, on="particle", how="inner")
    cand = cand[cand["r_um"].notna() & (cand["n_frames"] >= min_frames)]

    clean = clean_particle_set(labels)
    if clean:
        cand = cand[cand["particle"].isin(clean)]

    # Prefer round, in-focus beads if those quality columns are present.
    for col, thr in (("circ_resid_frac", 0.05), ("r_px_frame_cv", 0.10)):
        if col in cand.columns:
            keep = cand[cand[col] <= thr]
            if len(keep) >= 3:
                cand = keep

    # Reject D-vs-r outliers (robust MAD cut on k = D*r), so an anomalously
    # fast/slow bead can't masquerade as the representative small/mid/large one.
    if msd is not None and "D_um2_s" in msd.columns:
        cand = cand.merge(msd[["particle", "D_um2_s"]], on="particle",
                          how="left")
        cand = cand[cand["D_um2_s"].notna() & (cand["D_um2_s"] > 0)]
        k = cand["D_um2_s"] * cand["r_um"]
        med = k.median()
        mad = (k - med).abs().median()
        if mad > 0:
            keep = cand[(k - med).abs() <= 3.5 * mad]
            if len(keep) >= 3:
                cand = keep

    cand = cand.sort_values("r_um").reset_index(drop=True)
    if len(cand) < 3:
        raise SystemExit(f"need >=3 clean beads, found {len(cand)}")

    lo = cand.iloc[0]
    mid = cand.iloc[len(cand) // 2]
    hi = cand.iloc[-1]
    chosen = [(int(b.particle), float(b.r_um)) for b in (lo, mid, hi)]
    return chosen


def radius_lookup(radius, particle):
    row = radius.loc[radius["particle"] == particle, "r_um"]
    return float(row.iloc[0]) if len(row) and pd.notna(row.iloc[0]) else np.nan


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def fmt_val_err(val, err):
    """Format 'val +/- err' with two significant figures on the error."""
    if not np.isfinite(err) or err <= 0:
        return f"{val:.3f}", "0"
    ndec = min(4, max(3, -int(np.floor(np.log10(err))) + 1))
    return f"{val:.{ndec}f}", f"{err:.{ndec}f}"


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
        "lines.markersize": 4,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def make_figure(curves, run, T_label):
    """curves: list of dicts with keys
       particle, r_um, tau, msd, fit_mask, D, D_err, c, color, marker."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    for cv in curves:
        tau, msd = cv["tau"], cv["msd"]
        col, mk = cv["color"], cv["marker"]
        Ds, Des = fmt_val_err(cv["D"], cv["D_err"])
        label = (rf"$r={cv['r_um']:.2f}\,\mu$m,  "
                 rf"$D={Ds}\pm{Des}\,\mu$m$^2$/s")

        # ---- left: linear MSD + fit ----
        axL.plot(tau, msd, marker=mk, ls="none", color=col, ms=4.5,
                 alpha=0.9, label=label)
        tf = tau[cv["fit_mask"]]
        tline = np.linspace(0, tf.max(), 50)
        axL.plot(tline, 4 * cv["D"] * tline + cv["c"], "-", color=col, lw=1.6)

        # ---- right: log-log ----
        pos = msd > 0
        axR.plot(tau[pos], msd[pos], marker=mk, ls="none", color=col, ms=4.5,
                 alpha=0.9)

    # slope-1 reference on the log-log panel: a pure tau^1 power law anchored to
    # the middle curve's geometric centre, so it tracks the data where the
    # localisation-noise offset c no longer dominates (large tau).
    ref = curves[len(curves) // 2]
    rpos = ref["msd"] > 0
    t_ref, m_ref = ref["tau"][rpos], ref["msd"][rpos]
    i_anchor = len(t_ref) // 2
    t_anchor, m_anchor = t_ref[i_anchor], m_ref[i_anchor]
    tg = np.array([t_ref.min(), t_ref.max()])
    axR.plot(tg, m_anchor * (tg / t_anchor), "k--", lw=1.2,
             label=r"slope $=1$ ($\langle r^2\rangle\propto\tau$)")

    axL.set_xlabel(r"lag time  $\tau$  [s]")
    axL.set_ylabel(r"MSD  $\langle r^2\rangle$  [$\mu$m$^2$]")
    axL.set_xlim(left=0)
    axL.set_ylim(bottom=0)
    axL.set_title("Linear: $\\langle r^2\\rangle = 4D\\tau + c$")
    axL.legend(loc="upper left", fontsize=9)

    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel(r"lag time  $\tau$  [s]")
    axR.set_ylabel(r"MSD  $\langle r^2\rangle$  [$\mu$m$^2$]")
    axR.set_title("Log-log: slope $\\approx 1$")
    axR.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Brownian motion is normal diffusion  ({run}, {T_label})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run3")
    ap.add_argument("--beads", type=int, nargs="+", default=None,
                    help="explicit particle IDs (small mid large)")
    ap.add_argument("--min-frames", type=int, default=400,
                    help="min track length for auto-selection")
    ap.add_argument("--max-lag-s", type=float, default=5.0,
                    help="largest lag time to display [s]")
    ap.add_argument("--fit-lag-s", type=float, default=3.0,
                    help="largest lag time included in the linear fit [s]")
    ap.add_argument("--T", default="room T", help="temperature label for title")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    um_per_px = load_um_per_px()
    fps = load_fps(args.run)
    dt = 1.0 / fps
    px2um2 = um_per_px ** 2

    traj, radius, labels, msd = load_inputs(args.run)

    if args.beads:
        beads = [(p, radius_lookup(radius, p)) for p in args.beads]
        beads.sort(key=lambda b: (np.inf if np.isnan(b[1]) else b[1]))
    else:
        beads = chosen_beads(args.run, traj, radius, labels, msd,
                             args.min_frames)

    colors = ["#1f77b4", "#2ca02c", "#d62728"]   # small / mid / large
    markers = ["o", "s", "^"]
    max_lag_frames = int(round(args.max_lag_s * fps))

    curves = []
    print(f"\n{args.run}: um/px={um_per_px}, fps={fps:.3f}, dt={dt:.4f}s")
    print(f"{'particle':>9} {'r_um':>6} {'n_frames':>9} "
          f"{'D[um2/s]':>10} {'D_err':>9} {'c[um2]':>9}")
    for (p, r_um), col, mk in zip(beads, colors, markers):
        sub = traj.loc[traj["particle"] == p].sort_values("frame")
        lags, msd_px2, npairs = time_averaged_msd(
            sub["frame"].values, sub["x"].values, sub["y"].values,
            max_lag_frames)

        tau = lags * dt
        msd = msd_px2 * px2um2
        D, D_err, c, fmask = fit_linear_msd(tau, msd, npairs, args.fit_lag_s)

        curves.append(dict(particle=p, r_um=r_um, tau=tau, msd=msd,
                           fit_mask=fmask, D=D, D_err=D_err, c=c,
                           color=col, marker=mk))
        print(f"{p:>9} {r_um:>6.3f} {len(sub):>9} "
              f"{D:>10.4f} {D_err:>9.4f} {c:>9.4f}")

    set_style()
    fig = make_figure(curves, args.run, args.T)

    out = args.out or os.path.join(MEAS, args.run, "figures",
                                   "plot1_msd_vs_lag.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved -> {out}\n")


if __name__ == "__main__":
    main()
