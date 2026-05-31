#!/usr/bin/env python3
"""
bead_snapshots.py
=================
Extract real video snapshots of the representative single beads used in
Figure 1, as cropped thumbnails with a scale bar. Saves one PNG per bead plus a
combined 1xN strip, so the actual bead images can sit beside their MSD curves.

IMPORTANT -- needs the raw video, which is git-ignored and lives only in the
local  experiments/week1-system-calibration/videos/  folder. Run this on the
machine that has the videos (it is not runnable in the cloud session).

What it does
------------
1. Reads the run's trajectory (drift-subtracted positions) and the per-frame
   drift, so each bead can be located in the *raw* frame via
       raw_xy(f) = corrected_xy(f) + drift(f).
   (If your trajectory is NOT drift-subtracted, pass --drift-sign 0.)
2. For each bead, grabs a frame where the bead is present (default: its median
   frame), crops a fixed physical window centred on the bead, and renders a
   greyscale thumbnail with a scale bar and the radius label.
3. By default uses the same three beads as plot1 (auto-picked), or --beads.

Reader backends tried in order: pims, imageio (pyav), OpenCV.

Usage
-----
    python bead_snapshots.py --run run3
    python bead_snapshots.py --run run3 --beads 163 2 12
    python bead_snapshots.py --run run3 --video /path/to/run3.avi --window-um 12
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402

import plot1_msd_vs_lag as p1


# --------------------------------------------------------------------------- #
# Video reading (lazy; backends optional)
# --------------------------------------------------------------------------- #
def open_reader(path):
    if not os.path.exists(path):
        raise SystemExit(f"video not found: {path}\n"
                         "Pass --video PATH (raw videos are git-ignored).")
    try:
        import pims
        return ("pims", pims.open(path))
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        iio.imread(path, index=0, plugin="pyav")   # probe
        return ("iio", path)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return ("cv2", cap)
    except Exception:
        pass
    raise SystemExit("no working video backend (need pims, imageio[pyav], "
                     "or opencv-python)")


def get_frame(reader, f):
    kind, obj = reader
    if kind == "pims":
        img = np.asarray(obj[int(f)])
    elif kind == "iio":
        import imageio.v3 as iio
        img = iio.imread(obj, index=int(f), plugin="pyav")
    else:  # cv2
        import cv2
        obj.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = obj.read()
        if not ok:
            raise SystemExit(f"could not read frame {f}")
        img = img[..., ::-1]   # BGR -> RGB
    img = np.asarray(img)
    if img.ndim == 3:                       # to greyscale
        img = img[..., :3].mean(axis=2)
    return img


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_drift(run):
    d = p1.coherent_dir(run)
    p = os.path.join(d, "drift.csv")
    if not os.path.exists(p):
        p = os.path.join(os.path.dirname(d), "pipeline", "drift.csv")
    if os.path.exists(p):
        df = pd.read_csv(p).set_index("frame")
        return df["dx"].to_dict(), df["dy"].to_dict()
    return {}, {}


def pick_frame(sub, want):
    frames = sub["frame"].values
    if want == "median":
        target = np.median(frames)
    elif want == "first":
        target = frames.min()
    else:
        target = float(want)
    return int(frames[np.argmin(np.abs(frames - target))])


def draw_scalebar(ax, um_per_px, bar_um, npx):
    bar_px = bar_um / um_per_px
    x0 = npx * 0.06
    y0 = npx * 0.90
    ax.add_patch(Rectangle((x0, y0), bar_px, npx * 0.018,
                           color="white", ec="black", lw=0.4))
    ax.text(x0 + bar_px / 2, y0 - npx * 0.03, rf"{bar_um:g} $\mu$m",
            color="white", fontsize=8, ha="center", va="bottom")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run3")
    ap.add_argument("--beads", type=int, nargs="+", default=None)
    ap.add_argument("--video", default=None, help="path to the raw video")
    ap.add_argument("--frame", default="median",
                    help="'median' | 'first' | integer frame")
    ap.add_argument("--window-um", type=float, default=11.0,
                    help="crop window side length [um]")
    ap.add_argument("--scalebar-um", type=float, default=5.0)
    ap.add_argument("--drift-sign", type=float, default=1.0,
                    help="raw = corrected + sign*drift; use 0 to ignore drift")
    ap.add_argument("--min-frames", type=int, default=400)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    um_per_px = p1.load_um_per_px()
    traj, radius, labels, msd = p1.load_inputs(args.run)

    if args.beads:
        beads = [(p, p1.radius_lookup(radius, p)) for p in args.beads]
        beads.sort(key=lambda b: (np.inf if np.isnan(b[1]) else b[1]))
    else:
        beads = p1.pick_three_beads(traj, radius, labels, msd, args.min_frames)

    video = args.video or os.path.join(p1.ROOT, "videos",
                                       p1.VIDEO_KEY.get(args.run,
                                                        f"{args.run}.avi"))
    reader = open_reader(video)
    dx_map, dy_map = load_drift(args.run)

    half = (args.window_um / um_per_px) / 2.0
    npx = int(round(2 * half))
    outdir = args.outdir or os.path.join(p1.MEAS, args.run, "figures",
                                         "bead_snapshots")
    os.makedirs(outdir, exist_ok=True)

    p1.set_style()
    crops, infos = [], []
    print(f"\n{args.run}: video={video}  window={args.window_um}um "
          f"({npx}px)  drift_sign={args.drift_sign}")
    for p, r_um in beads:
        sub = traj.loc[traj["particle"] == p].sort_values("frame")
        f = pick_frame(sub, args.frame)
        row = sub.loc[sub["frame"] == f].iloc[0]
        cx = row["x"] + args.drift_sign * dx_map.get(f, 0.0)
        cy = row["y"] + args.drift_sign * dy_map.get(f, 0.0)

        img = get_frame(reader, f)
        H, W = img.shape
        x0 = int(round(np.clip(cx - half, 0, W - npx)))
        y0 = int(round(np.clip(cy - half, 0, H - npx)))
        crop = img[y0:y0 + npx, x0:x0 + npx]
        crops.append(crop)
        infos.append((p, r_um, f))
        print(f"  p{p}: r={r_um:.2f}um  frame={f}  "
              f"raw_xy=({cx:.0f},{cy:.0f})  crop={crop.shape}")

        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.imshow(crop, cmap="gray", origin="upper")
        draw_scalebar(ax, um_per_px, args.scalebar_um, npx)
        ax.set_title(rf"p{p}, $r={r_um:.2f}\,\mu$m", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = os.path.join(outdir, f"bead_p{p}.png")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)

    # combined strip
    nb = len(crops)
    fig, axes = plt.subplots(1, nb, figsize=(2.6 * nb, 2.8))
    axes = np.atleast_1d(axes)
    for ax, crop, (p, r_um, f) in zip(axes, crops, infos):
        ax.imshow(crop, cmap="gray", origin="upper")
        draw_scalebar(ax, um_per_px, args.scalebar_um, crop.shape[0])
        ax.set_title(rf"$r={r_um:.2f}\,\mu$m", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    strip = os.path.join(outdir, "bead_strip.png")
    fig.savefig(strip, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nsaved {nb} crops + strip -> {outdir}\n")


if __name__ == "__main__":
    main()
