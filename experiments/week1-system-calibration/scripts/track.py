"""
track.py  (week1-system-calibration)
------------------------------------
Particle tracking with trackpy (Crocker-Grier), video -> trajectory.csv.

Pipeline:
  locate/batch  : find bright particle centres each frame (sub-pixel)
  link          : join detections across frames into trajectories
  filter_stubs  : drop short spurious tracks
  drift-subtract: remove collective flow so MSD reflects diffusion, not drift
                  (this is the booklet's "is the system isotropic / is there a
                   drift?" check, handled in analysis)

TWO MODES
  --tune   locate on ONE frame, save an annotated overlay + a mass histogram +
           a sub-pixel-bias plot, and print the feature count. Use these to set
           --diameter (odd int, a bit larger than the particle) and --minmass
           (cut off the noise/defocus peak). Nothing is linked in this mode.
  (full)   batch over all frames, link, filter, drift-subtract, write
           measurements/<stem>/trajectory.csv (+ drift.csv, +preview plots).

Calibration & time base come from this week's calibration/scale.json and
videos_meta.json via _paths, so D later comes out in physical units.

Usage
-----
    cd experiments/week1-system-calibration
    # 1) tune on a real (1632) run:
    python scripts/track.py run2.avi --tune --diameter 11 --frame 0
    # 2) once diameter/minmass look right, track the whole clip:
    python scripts/track.py run2.avi --diameter 11 --minmass 200 --search 15 --memory 3
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths


def gray_frames(path, max_frames=None):
    """Yield grayscale uint8 frames from an AVI/MP4 via imageio+pyav (cv2 fallback)."""
    try:
        import imageio.v3 as iio
        for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
            if max_frames and i >= max_frames:
                return
            yield _gray(fr)
        return
    except Exception as e:
        print(f"[imageio/pyav failed: {e}] using OpenCV", file=sys.stderr)
    import cv2
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        yield _gray(fr)
        i += 1
    cap.release()


def _gray(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    return a.astype(np.uint8)


def nth_frame(path, n):
    for i, fr in enumerate(gray_frames(path)):
        if i == n:
            return fr
    raise IndexError(f"frame {n} not found in {path}")


def main():
    ap = argparse.ArgumentParser(description="Track particles in a Brownian clip.")
    ap.add_argument("video", help="clip name in videos/ or a path")
    ap.add_argument("--diameter", type=int, default=11, help="trackpy feature diameter (ODD px)")
    ap.add_argument("--minmass", type=float, default=None, help="min integrated brightness")
    ap.add_argument("--invert", action="store_true", help="track DARK features instead of bright")
    ap.add_argument("--search", type=float, default=15, help="link search range (px/frame)")
    ap.add_argument("--memory", type=int, default=3, help="frames a particle may vanish")
    ap.add_argument("--stub", type=int, default=50, help="drop tracks shorter than this many frames")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tune", action="store_true", help="locate one frame only; save diagnostics")
    ap.add_argument("--frame", type=int, default=0, help="frame index for --tune")
    args = ap.parse_args()

    import trackpy as tp

    path = _paths.video(args.video)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = _paths.clip_dir(stem)
    os.makedirs(out, exist_ok=True)

    if args.diameter % 2 == 0:
        sys.exit("--diameter must be ODD (trackpy requirement)")

    # ---------------- TUNE: locate on one frame, save diagnostics ----------
    if args.tune:
        frame = nth_frame(path, args.frame)
        f = tp.locate(frame, args.diameter, minmass=args.minmass, invert=args.invert)
        tdir = os.path.join(out, "tune")
        os.makedirs(tdir, exist_ok=True)
        print(f"[tune] {stem} frame {args.frame}: {len(f)} features "
              f"(diameter={args.diameter}, minmass={args.minmass})")

        plt.figure(figsize=(10, 7.5))
        tp.annotate(f, frame, plot_style={"markersize": 6})
        plt.title(f"{stem} f{args.frame}: {len(f)} features, d={args.diameter}")
        plt.savefig(os.path.join(tdir, "annotated.png"), dpi=130, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(f["mass"], bins=50)
        plt.xlabel("mass (integrated brightness)"); plt.ylabel("count")
        plt.title("set --minmass to cut the low-mass noise/defocus peak")
        plt.savefig(os.path.join(tdir, "mass_hist.png"), dpi=120, bbox_inches="tight")
        plt.close()

        # sub-pixel bias: the decimal part of x,y positions should be ~flat.
        # A U-shape or central peak => --diameter is too small. (Version-
        # independent: we histogram x%1 and y%1 ourselves rather than rely on
        # tp.subpx_bias, whose signature varies across trackpy releases.)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].hist(f["x"] % 1, bins=20); ax[0].set_title("x mod 1")
        ax[1].hist(f["y"] % 1, bins=20); ax[1].set_title("y mod 1")
        fig.suptitle("sub-pixel bias (flat = good; peaked/U => raise --diameter)")
        fig.savefig(os.path.join(tdir, "subpx_bias.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"[tune] saved annotated.png, mass_hist.png, subpx_bias.png -> {tdir}")
        print("       send these back; then re-run without --tune to track the clip.")
        return

    # ---------------- FULL: locate (streaming) -> link -> filter -> drift --
    # tp.batch wants an indexable sequence; we instead locate frame-by-frame
    # from the generator so only ONE frame is in RAM at a time (the long runs
    # are ~1600-1900 frames of 1632x1224 -> a full list would be 3-4 GB).
    import pandas as pd
    print(f"[track] {stem}: locating features frame-by-frame (streaming)...")
    tp.quiet()
    parts = []
    n_done = 0
    for i, frame in enumerate(gray_frames(path, args.max_frames)):
        f = tp.locate(frame, args.diameter, minmass=args.minmass, invert=args.invert)
        if len(f):
            f["frame"] = i
            parts.append(f)
        n_done = i + 1
        if n_done % 100 == 0:
            print(f"    ...located {n_done} frames")
    if not parts:
        sys.exit("No features found in any frame \u2014 check --minmass/--diameter.")
    feats = pd.concat(parts, ignore_index=True)
    print(f"[track] {len(feats)} detections over "
          f"{feats['frame'].nunique()} frames; linking...")

    traj = tp.link(feats, search_range=args.search, memory=args.memory)
    n0 = traj["particle"].nunique()
    traj = tp.filter_stubs(traj, args.stub)
    n1 = traj["particle"].nunique()
    print(f"[track] {n0} trajectories -> {n1} after dropping stubs < {args.stub} frames")

    # drift: collective motion shared by all particles
    drift = tp.compute_drift(traj)
    traj_nd = tp.subtract_drift(traj.copy(), drift)
    drift.to_csv(os.path.join(out, "drift.csv"))

    traj_nd.to_csv(os.path.join(out, "trajectory.csv"), index=False)
    print(f"[track] saved trajectory.csv ({n1} particles, drift-subtracted) -> {out}")

    # quick previews
    plt.figure(figsize=(6, 3))
    plt.plot(drift.index, drift["x"], label="drift x")
    plt.plot(drift.index, drift["y"], label="drift y")
    plt.xlabel("frame"); plt.ylabel("cumulative drift (px)"); plt.legend()
    plt.title(f"{stem}: ensemble drift (subtracted before MSD)")
    plt.savefig(os.path.join(out, "drift.png"), dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5.3))
    tp.plot_traj(traj_nd)
    plt.title(f"{stem}: {n1} drift-subtracted trajectories")
    plt.savefig(os.path.join(out, "trajectories.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[track] saved drift.png, trajectories.png -> {out}")


if __name__ == "__main__":
    main()
