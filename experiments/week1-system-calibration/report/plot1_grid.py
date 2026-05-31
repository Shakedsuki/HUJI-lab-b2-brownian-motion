#!/usr/bin/env python3
"""
plot1_grid.py
=============
Report figure 1, multi-run version -- a 2x2 grid of MSD-vs-lag panels, one per
run, for visually comparing normal diffusion across runs (and temperatures).

Each panel reuses the plot1 machinery: three clean single beads spanning the
run's radius range, each bead's gap-aware time-averaged 2D MSD fit to
<r^2> = 4D*tau + c. The fitted D (with error) is shown in the per-panel legend.

Usage
-----
    python plot1_grid.py                       # runs 3,4,5,6
    python plot1_grid.py --runs run3 run7 run9 # any set (filled row-major)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import plot1_msd_vs_lag as p1


COLORS = ["#1f77b4", "#2ca02c", "#d62728"]   # small / mid / large radius
MARKERS = ["o", "s", "^"]


def build_curves(run, min_frames, max_lag_s, fit_lag_s, beads=None):
    """Return (list-of-curve-dicts, info-string) for one run."""
    um_per_px = p1.load_um_per_px()
    fps = p1.load_fps(run)
    dt = 1.0 / fps
    px2um2 = um_per_px ** 2

    traj, radius, labels, msd = p1.load_inputs(run)
    if beads:
        chosen = [(b, p1.radius_lookup(radius, b)) for b in beads]
        chosen.sort(key=lambda t: (np.inf if np.isnan(t[1]) else t[1]))
    else:
        chosen = p1.pick_three_beads(traj, radius, labels, msd, min_frames)

    max_lag_frames = int(round(max_lag_s * fps))
    curves = []
    for (p, r_um), col, mk in zip(chosen, COLORS, MARKERS):
        sub = traj.loc[traj["particle"] == p].sort_values("frame")
        lags, msd_px2, npairs = p1.time_averaged_msd(
            sub["frame"].values, sub["x"].values, sub["y"].values,
            max_lag_frames)
        tau = lags * dt
        msd = msd_px2 * px2um2
        D, D_err, c, fmask = p1.fit_linear_msd(tau, msd, npairs, fit_lag_s)
        curves.append(dict(particle=p, r_um=r_um, tau=tau, msd=msd,
                           fit_mask=fmask, D=D, D_err=D_err, c=c,
                           color=col, marker=mk))
    return curves, f"fps={fps:.2f}"


def draw_panel(ax, curves, run):
    for cv in curves:
        Ds, Des = p1.fmt_val_err(cv["D"], cv["D_err"])
        label = (rf"$r={cv['r_um']:.2f}\,\mu$m: "
                 rf"$D={Ds}\pm{Des}$")
        ax.plot(cv["tau"], cv["msd"], cv["marker"], ls="none",
                color=cv["color"], ms=4, alpha=0.9, label=label)
        tf = cv["tau"][cv["fit_mask"]]
        tline = np.linspace(0, tf.max(), 50)
        ax.plot(tline, 4 * cv["D"] * tline + cv["c"], "-",
                color=cv["color"], lw=1.5)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(run, fontsize=12)
    ax.legend(loc="upper left", fontsize=8, title=r"$D$ in $\mu$m$^2$/s",
              title_fontsize=8)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+",
                    default=["run3", "run4", "run6"])
    ap.add_argument("--min-frames", type=int, default=400)
    ap.add_argument("--max-lag-s", type=float, default=5.0)
    ap.add_argument("--fit-lag-s", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n = len(args.runs)
    ncol = n if n <= 3 else 2     # <=3 runs in one row, else a 2-wide grid
    nrow = int(np.ceil(n / ncol))

    p1.set_style()
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow),
                             squeeze=False)
    flat = axes.ravel()

    for ax, run in zip(flat, args.runs):
        curves, info = build_curves(run, args.min_frames,
                                    args.max_lag_s, args.fit_lag_s)
        draw_panel(ax, curves, run)
        print(f"{run}: " + "  ".join(
            f"p{cv['particle']}(r={cv['r_um']:.2f},D={cv['D']:.3f})"
            for cv in curves) + f"  [{info}]")

    for ax in flat[n:]:
        ax.axis("off")

    # shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel(r"lag time  $\tau$  [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"MSD  $\langle r^2\rangle$  [$\mu$m$^2$]")

    fig.suptitle(r"MSD vs lag time: $\langle r^2\rangle = 4D\tau + c$ "
                 r"(three beads per run)", fontsize=13, y=1.0)
    fig.tight_layout()

    tag = "-".join(r.replace("run", "") for r in args.runs)
    out = args.out or os.path.join(p1.MEAS, os.pardir, "figures",
                                   f"plot1_grid_runs{tag}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved -> {out}\n")


if __name__ == "__main__":
    main()
