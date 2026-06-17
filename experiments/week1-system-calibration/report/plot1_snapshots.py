#!/usr/bin/env python3
"""
plot1_snapshots.py
==================
Figure 1 (MSD-vs-lag grid) combined with the bead snapshot thumbnails, in two
layouts so they can be compared:

    --layout strip   MSD panels on top, a row of the run's three bead
                     thumbnails directly beneath each panel.
    --layout inset   thumbnails placed as small inset images inside each MSD
                     panel (stacked at the right edge), framed in the matching
                     curve colour.

The thumbnails are the raw crops written by bead_snapshots.py
(measurements/<run>/figures/bead_snapshots/bead_p<ID>_raw.png). If a crop is
missing you can pass --placeholder to synthesize a stand-in microsphere image
from the bead radius -- useful for previewing the layout before the real crops
(which need the local video) exist.

Usage
-----
    # preview both layouts with synthetic thumbnails (no video needed)
    python plot1_snapshots.py --layout strip  --placeholder
    python plot1_snapshots.py --layout inset  --placeholder

    # once bead_snapshots.py has produced the real crops locally
    python plot1_snapshots.py --layout strip
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import plot1_msd_vs_lag as p1
import plot1_grid as pg


# --------------------------------------------------------------------------- #
# Thumbnail loading / synthesis
# --------------------------------------------------------------------------- #
def crop_path(run, particle):
    return os.path.join(p1.MEAS, run, "figures", "bead_snapshots",
                        f"bead_p{particle}_raw.png")


def synth_bead(r_um, um_per_px, window_um=11.0, seed=0):
    """Synthetic phase-contrast-like microsphere (placeholder only)."""
    npx = int(round(window_um / um_per_px))
    r_px = max(2.0, r_um / um_per_px)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:npx, 0:npx]
    rr = np.hypot(xx - npx / 2, yy - npx / 2)
    img = 0.55 + 0.03 * rng.standard_normal((npx, npx))
    img += 0.33 * np.exp(-(rr / (0.62 * r_px)) ** 2)         # bright core
    img -= 0.32 * np.exp(-((rr - r_px) / (0.16 * r_px)) ** 2)  # dark ring
    img += 0.06 * np.exp(-((rr - 1.25 * r_px) / (0.30 * r_px)) ** 2)
    return np.clip(img, 0, 1)


def load_thumb(run, particle, r_um, um_per_px, placeholder):
    path = crop_path(run, particle)
    if os.path.exists(path):
        img = plt.imread(path)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        return img, False
    if placeholder:
        return synth_bead(r_um, um_per_px, seed=particle), True
    raise SystemExit(
        f"missing crop: {path}\nRun bead_snapshots.py for {run} first, "
        f"or pass --placeholder to preview the layout.")


def frame_axes(ax, color, lw=2.2):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(color)
        s.set_linewidth(lw)
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #
def layout_strip(runs, allcurves, thumbs, placeholder):
    nrun = len(runs)
    fig = plt.figure(figsize=(4.8 * nrun, 5.7))
    gs = fig.add_gridspec(2, nrun, height_ratios=[3.0, 1.15], hspace=0.30,
                          wspace=0.22)
    for j, run in enumerate(runs):
        axm = fig.add_subplot(gs[0, j])
        pg.draw_panel(axm, allcurves[run], run)
        axm.set_xlabel(r"lag time  $\tau$  [s]")
        if j == 0:
            axm.set_ylabel(r"MSD  $\langle r^2\rangle$  [$\mu$m$^2$]")

        sub = gs[1, j].subgridspec(1, 3, wspace=0.08)
        for i, cv in enumerate(allcurves[run]):
            axt = fig.add_subplot(sub[0, i])
            img, _ = thumbs[(run, cv["particle"])]
            axt.imshow(img, cmap="gray", origin="upper")
            frame_axes(axt, cv["color"])
            axt.set_title(rf"$r={cv['r_um']:.2f}$", fontsize=8, color=cv["color"],
                          pad=2)
    return fig


def layout_inset(runs, allcurves, thumbs, placeholder):
    nrun = len(runs)
    fig, axes = plt.subplots(1, nrun, figsize=(5.0 * nrun, 4.6), squeeze=False)
    for ax, run in zip(axes[0], runs):
        pg.draw_panel(ax, allcurves[run], run)
        ax.set_xlabel(r"lag time  $\tau$  [s]")
        if ax is axes[0, 0]:
            ax.set_ylabel(r"MSD  $\langle r^2\rangle$  [$\mu$m$^2$]")
        # stack the three thumbnails down the right edge
        ys = [0.66, 0.35, 0.04]
        for cv, y in zip(allcurves[run], ys):
            iax = ax.inset_axes([0.80, y, 0.18, 0.30])
            img, _ = thumbs[(run, cv["particle"])]
            iax.imshow(img, cmap="gray", origin="upper")
            frame_axes(iax, cv["color"], lw=1.8)
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["run3", "run4", "run6"])
    ap.add_argument("--layout", choices=["strip", "inset"], default="strip")
    ap.add_argument("--placeholder", action="store_true",
                    help="synthesize stand-in thumbnails for any missing crop")
    ap.add_argument("--min-frames", type=int, default=400)
    ap.add_argument("--max-lag-s", type=float, default=5.0)
    ap.add_argument("--fit-lag-s", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    um_per_px = p1.load_um_per_px()

    allcurves, thumbs = {}, {}
    n_synth = 0
    for run in args.runs:
        curves, _ = pg.build_curves(run, args.min_frames, args.max_lag_s,
                                    args.fit_lag_s)
        allcurves[run] = curves
        for cv in curves:
            img, synth = load_thumb(run, cv["particle"], cv["r_um"],
                                    um_per_px, args.placeholder)
            thumbs[(run, cv["particle"])] = (img, synth)
            n_synth += int(synth)

    pg.p1.set_style() if hasattr(pg, "p1") else p1.set_style()
    if args.layout == "strip":
        fig = layout_strip(args.runs, allcurves, thumbs, args.placeholder)
    else:
        fig = layout_inset(args.runs, allcurves, thumbs, args.placeholder)
    fig.tight_layout()

    tag = "-".join(r.replace("run", "") for r in args.runs)
    suffix = "_PLACEHOLDER" if n_synth else ""
    out = os.path.abspath(args.out or os.path.join(
        p1.ROOT, "figures", f"plot1_snapshots_{args.layout}_runs{tag}{suffix}.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    try:
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
    except PermissionError:
        pass
    note = f"  ({n_synth} synthetic placeholder thumbnails)" if n_synth else ""
    print(f"saved -> {out}{note}")


if __name__ == "__main__":
    main()
