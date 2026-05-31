#!/usr/bin/env python3
"""
plot1_with_wall.py
==================
Report figure 1, wall variant -- the standard small/mid/large MSD-vs-lag plot,
with the wall-hindered (large-radius, r > r*) beads overlaid as extra,
distinguished curves.

The buoyant polyethylene spheres above the sedimentation scale r* drift up to
the top coverslip; wall drag suppresses their diffusion, so their MSD slope
sits below what their size alone would give. Overlaying them on the normal
small/mid/large illustration shows that population directly. The original
plot1 figure is left untouched -- this writes a separate PNG.

Reuses the plot1 primitives (loaders, MSD, fit) and plot2's r* helper.

Usage
-----
    python plot1_with_wall.py                 # run3
    python plot1_with_wall.py --run run3 --n-wall 3
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import plot1_msd_vs_lag as p1
import plot2_D_vs_inv_r as p2     # sediment_r_star_um


BASE_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]   # small / mid / large (free set)
BASE_MARKERS = ["o", "s", "^"]
WALL_COLORS = ["#7b6a52", "#a07e10", "#5b4636"]   # earthy tones for wall beads
WALL_MARKERS = ["D", "v", "P"]


def msd_curve(traj, particle, r_um, fps, um_per_px, max_lag_s, fit_lag_s):
    dt = 1.0 / fps
    sub = traj.loc[traj["particle"] == particle].sort_values("frame")
    lags, msd_px2, npairs = p1.time_averaged_msd(
        sub["frame"].values, sub["x"].values, sub["y"].values,
        int(round(max_lag_s * fps)))
    tau = lags * dt
    msd = msd_px2 * um_per_px ** 2
    D, D_err, c, fmask = p1.fit_linear_msd(tau, msd, npairs, fit_lag_s)
    return dict(particle=particle, r_um=r_um, tau=tau, msd=msd,
                fit_mask=fmask, D=D, D_err=D_err, c=c)


def pick_wall_beads(traj, radius, labels, r_star, min_frames, n, exclude):
    """Up to n clean wall-hindered beads (r > r*), spread across radius."""
    counts = traj.groupby("particle").size().rename("n_frames")
    cand = radius.merge(counts, on="particle", how="inner")
    cand = cand[cand["r_um"].notna() & (cand["n_frames"] >= min_frames)
                & (cand["r_um"] > r_star) & ~cand["particle"].isin(exclude)]
    clean = p1.clean_particle_set(labels)
    if clean:
        cand = cand[cand["particle"].isin(clean)]
    for col, thr in (("circ_resid_frac", 0.05), ("r_px_frame_cv", 0.10)):
        if col in cand.columns:
            keep = cand[cand[col] <= thr]
            if len(keep) >= n:
                cand = keep
    cand = cand.sort_values("r_um").reset_index(drop=True)
    if len(cand) == 0:
        return []
    idx = np.unique(np.linspace(0, len(cand) - 1, min(n, len(cand)))
                    .round().astype(int))
    return [(int(cand.loc[i, "particle"]), float(cand.loc[i, "r_um"]))
            for i in idx]


def draw(ax_lin, ax_log, base, wall, r_star):
    # --- base small/mid/large (free illustration) ---
    for cv, col, mk in zip(base, BASE_COLORS, BASE_MARKERS):
        Ds, Des = p1.fmt_val_err(cv["D"], cv["D_err"])
        lab = rf"$r={cv['r_um']:.2f}\,\mu$m: $D={Ds}\pm{Des}$"
        ax_lin.plot(cv["tau"], cv["msd"], mk, ls="none", color=col, ms=4.5,
                    alpha=0.9, label=lab)
        tf = cv["tau"][cv["fit_mask"]]
        tl = np.linspace(0, tf.max(), 50)
        ax_lin.plot(tl, 4 * cv["D"] * tl + cv["c"], "-", color=col, lw=1.6)
        pos = cv["msd"] > 0
        ax_log.plot(cv["tau"][pos], cv["msd"][pos], mk, ls="none", color=col,
                    ms=4.5, alpha=0.9)

    # --- wall-hindered overlay (open markers, dashed fit) ---
    for cv, col, mk in zip(wall, WALL_COLORS, WALL_MARKERS):
        Ds, Des = p1.fmt_val_err(cv["D"], cv["D_err"])
        lab = rf"wall $r={cv['r_um']:.2f}\,\mu$m: $D={Ds}\pm{Des}$"
        ax_lin.plot(cv["tau"], cv["msd"], mk, ls="none", mfc="none", mec=col,
                    ms=4.5, alpha=0.9, label=lab)
        tf = cv["tau"][cv["fit_mask"]]
        tl = np.linspace(0, tf.max(), 50)
        ax_lin.plot(tl, 4 * cv["D"] * tl + cv["c"], "--", color=col, lw=1.4)
        pos = cv["msd"] > 0
        ax_log.plot(cv["tau"][pos], cv["msd"][pos], mk, ls="none", mfc="none",
                    mec=col, ms=4.5, alpha=0.9)

    # slope-1 guide on log-log, anchored to the mid base curve
    ref = base[len(base) // 2]
    rp = ref["msd"] > 0
    t, m = ref["tau"][rp], ref["msd"][rp]
    ia = len(t) // 2
    tg = np.array([t.min(), t.max()])
    ax_log.plot(tg, m[ia] * (tg / t[ia]), "k--", lw=1.1,
                label=r"slope $=1$")

    ax_lin.set_xlabel(r"lag time  $\tau$  [s]")
    ax_lin.set_ylabel(r"MSD  $\langle r^2\rangle$  [$\mu$m$^2$]")
    ax_lin.set_xlim(left=0)
    ax_lin.set_ylim(bottom=0)
    ax_lin.set_title("Linear: $\\langle r^2\\rangle = 4D\\tau + c$")
    ax_lin.legend(loc="upper left", fontsize=8)

    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_xlabel(r"lag time  $\tau$  [s]")
    ax_log.set_ylabel(r"MSD  $\langle r^2\rangle$  [$\mu$m$^2$]")
    ax_log.set_title(rf"Log-log (wall beads: $r>r^*={r_star:.2f}\,\mu$m)")
    ax_log.legend(loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run3")
    ap.add_argument("--n-wall", type=int, default=3)
    ap.add_argument("--min-frames", type=int, default=400)
    ap.add_argument("--max-lag-s", type=float, default=5.0)
    ap.add_argument("--fit-lag-s", type=float, default=3.0)
    ap.add_argument("--T", type=float, default=25.0)
    ap.add_argument("--delta-rho", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    um_per_px = p1.load_um_per_px()
    fps = p1.load_fps(args.run)
    r_star = p2.sediment_r_star_um(args.T, args.delta_rho)

    traj, radius, labels, msd = p1.load_inputs(args.run)
    base_beads = p1.pick_three_beads(traj, radius, labels, msd, args.min_frames)
    base_ids = {p for p, _ in base_beads}
    wall_beads = pick_wall_beads(traj, radius, labels, r_star,
                                 args.min_frames, args.n_wall, base_ids)

    base = [msd_curve(traj, p, r, fps, um_per_px, args.max_lag_s, args.fit_lag_s)
            for p, r in base_beads]
    wall = [msd_curve(traj, p, r, fps, um_per_px, args.max_lag_s, args.fit_lag_s)
            for p, r in wall_beads]

    print(f"\n{args.run}: r*={r_star:.2f}um  fps={fps:.2f}")
    print("base (small/mid/large): " + "  ".join(
        f"p{c['particle']}(r={c['r_um']:.2f},D={c['D']:.3f})" for c in base))
    print("wall (r>r*):            " + "  ".join(
        f"p{c['particle']}(r={c['r_um']:.2f},D={c['D']:.3f})" for c in wall))

    p1.set_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    draw(axL, axR, base, wall, r_star)
    fig.suptitle(f"MSD vs lag time with wall-hindered beads  ({args.run})",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = args.out or os.path.join(p1.MEAS, args.run, "figures",
                                   "plot1_with_wall.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved -> {out}\n")


if __name__ == "__main__":
    main()
