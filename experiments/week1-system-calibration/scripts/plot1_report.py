"""
plot1_report.py  (week1-system-calibration)
-------------------------------------------
The reportable version of Plot 1 (booklet Part 1): a handful of CLEAN beads'
mean-squared displacement vs lag time, showing directly that

        <r^2>(tau) = 4 D tau + c        (2D projection => factor 4)

is LINEAR in tau (normal diffusion), with per-bead D from the short-lag slope.
Two panels: linear axes (the straight-line "money" plot) + log-log (slope ~ 1).

Beads are auto-picked from msd.csv as the cleanest available (long, in-focus,
round, near-zero intercept) and chosen to SPAN a range of D so the differing
slopes preview the size dependence (Plot 2). Override with --beads.

This recomputes the time-averaged MSD directly from trajectory.csv (no trackpy
dependency) so it is self-contained.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/plot1_report.py run3 --tag d21m600
    python scripts/plot1_report.py run3                       # untagged dir
    python scripts/plot1_report.py run3 --beads 28 24 297     # explicit beads
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


def tamsd(sub, mpp, dt, max_lag):
    """Time-averaged MSD (um^2) vs lag time (s) for one bead, gap-aware."""
    sub = sub.sort_values("frame")
    f = sub["frame"].values.astype(int)
    x = sub["x"].values * mpp
    y = sub["y"].values * mpp
    pos = {fr: (xx, yy) for fr, xx, yy in zip(f, x, y)}
    lags, msd = [], []
    for lag in range(1, max_lag + 1):
        sq = [(pos[fr + lag][0] - pos[fr][0]) ** 2 + (pos[fr + lag][1] - pos[fr][1]) ** 2
              for fr in f if fr + lag in pos]
        if len(sq) >= 5:
            lags.append(lag * dt)
            msd.append(float(np.mean(sq)))
    return np.array(lags), np.array(msd)


def pick_clean(msd_csv, n):
    """Auto-pick n clean beads spanning the D range."""
    m = pd.read_csv(msd_csv)
    gates = [
        dict(nf=400, cv=0.10, ec=0.20, ic=0.05),   # strict
        dict(nf=200, cv=0.15, ec=0.25, ic=0.10),   # relaxed fallback
    ]
    for g in gates:
        c = m[(m.n_frames >= g["nf"]) & (m.size_cv < g["cv"]) &
              (m.ecc_med < g["ec"]) & (m.intercept_um2.abs() < g["ic"])]
        if len(c) >= n:
            break
    if len(c) == 0:
        c = m.sort_values("n_frames", ascending=False).head(n)
    c = c.sort_values("D_um2_s")
    # spread across D: take min, max, and evenly spaced in between
    idx = np.linspace(0, len(c) - 1, n).astype(int)
    return c.iloc[idx]["particle"].astype(int).tolist()


def main():
    ap = argparse.ArgumentParser(description="Reportable Plot 1: clean-bead MSD vs t.")
    ap.add_argument("run", help="run stem, e.g. run3")
    ap.add_argument("--tag", default=None, help="measurements/<run>/<tag>/ (match track.py)")
    ap.add_argument("--beads", type=int, nargs="*", default=None,
                    help="explicit particle ids; default = auto-pick cleanest")
    ap.add_argument("--n-show", type=int, default=3, help="how many beads to feature")
    ap.add_argument("--fit-lag", type=int, default=20, help="fit slope over lags <= this (frames)")
    ap.add_argument("--max-lag", type=int, default=55, help="display MSD up to this lag (frames)")
    args = ap.parse_args()

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    tcsv = os.path.join(cdir, "trajectory.csv")
    mcsv = os.path.join(cdir, "msd.csv")
    if not os.path.exists(tcsv):
        sys.exit(f"no trajectory.csv in {cdir} -- run track.py first")

    mpp = _paths.load_scale() or 1.0
    meta_video = _paths.load_runs().get("runs", {}).get(stem, {}).get("video", stem + ".avi")
    fps = _paths.fps_of(meta_video) or 9.30
    dt = 1.0 / fps

    beads = args.beads
    if not beads:
        if not os.path.exists(mcsv):
            sys.exit("no msd.csv to auto-pick beads; run msd_fit.py or pass --beads")
        beads = pick_clean(mcsv, args.n_show)
    print(f"[plot1] {stem}: mpp={mpp} um/px, fps={fps:.3f}; beads {beads}")

    traj = pd.read_csv(tcsv)
    figure_style.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = ["C0", "C1", "C3", "C2", "C4", "C5"]
    labels = ["small bead", "medium bead", "large bead"] + [f"bead {b}" for b in beads[3:]]
    guide = None

    for k, pid in enumerate(beads):
        sub = traj[traj["particle"] == pid]
        lags, msd = tamsd(sub, mpp, dt, args.max_lag)
        if len(lags) < 3:
            continue
        guide = lags
        fm = lags <= args.fit_lag * dt
        (sl, ic), cov = np.polyfit(lags[fm], msd[fm], 1, cov=True)
        D, Derr = sl / 4.0, float(np.sqrt(cov[0, 0])) / 4.0
        col = colors[k % len(colors)]
        lab = labels[k] if k < len(labels) else f"bead {pid}"
        ax[0].plot(lags, msd, "o", ms=3, color=col, alpha=0.6)
        xs = np.linspace(0, lags[fm].max(), 50)
        ax[0].plot(xs, sl * xs + ic, "-", color=col, lw=1.8,
                   label=f"{lab}: D = {D:.3f} $\\pm$ {Derr:.4f} " + r"$\mu$m$^2$/s")
        ax[1].loglog(lags, msd, "o", ms=3, color=col, alpha=0.6, label=lab)

    ax[0].set_xlabel(r"lag time  $\tau$  [s]")
    ax[0].set_ylabel(r"$\langle r^2 \rangle$   [$\mu$m$^2$]")
    ax[0].set_title(r"MSD linear in $t$   (fit $\langle r^2\rangle = 4D\tau + c$,  "
                    rf"$\tau \leq {args.fit_lag/fps:.1f}$ s)", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_xlim(0, args.max_lag * dt)

    if guide is not None:
        gx = np.array([guide.min(), guide.max() * 0.5])
        ax[1].loglog(gx, gx / gx[0] * 0.05, "k--", lw=1, label="slope 1")
    ax[1].set_xlabel(r"lag time  $\tau$  [s]")
    ax[1].set_ylabel(r"$\langle r^2 \rangle$   [$\mu$m$^2$]")
    ax[1].set_title(r"log$-$log: slope $\approx$ 1  $\Rightarrow$  normal diffusion", fontsize=10)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    name = f"plot1_report_{stem}{'_' + args.tag if args.tag else ''}.png"
    path = figure_style.savefig(name, fig=fig)
    plt.close(fig)
    print(f"[plot1] wrote {path}")


if __name__ == "__main__":
    main()
