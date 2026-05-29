"""
analysis/inspect_video.py
==========================
HUJI Lab B2 — Brownian Motion, 2026
Shaked Sukiennik & Nir Cohen

FIRST computational step on a raw measurement video. Does NOT track anything —
it just tells us what we're working with so we can tune the tracker:

  - prints metadata: n_frames, resolution, dtype, fps (if available)
  - saves N evenly-spaced sample frames as PNG  -> for setting trackpy `diameter`
  - saves a MAX / MIN / STD projection over all frames:
        * MAX  -> brightest-pixel hold; a single Brownian particle paints its
                  whole trajectory -> instant sanity check of the walk
        * STD  -> pixels that move light up -> isolates moving particle(s) from
                  static dirt/illumination gradient; reveals if >1 particle or
                  a static speck we might mistrack
  - reports mean intensity per frame (a crude drift/defocus indicator)

Supports: .avi / .mp4 (imageio-ffmpeg), .tif/.tiff stacks (tifffile),
or a directory of image frames.

Usage
-----
    python analysis/inspect_video.py path/to/video.avi
    python analysis/inspect_video.py path/to/frames_dir/  --samples 8
    python analysis/inspect_video.py vid.avi --out analysis/inspect_out --max-frames 600
"""

import argparse
import os
import sys
import glob
import numpy as np


def load_frames(path, max_frames=None):
    """Return (frames_iter_or_array, n_frames, fps). Grayscale float32 frames."""
    fps = None

    # --- directory of image files -------------------------------------------
    if os.path.isdir(path):
        import imageio.v3 as iio
        files = sorted(
            f for ext in ("png", "tif", "tiff", "jpg", "jpeg", "bmp")
            for f in glob.glob(os.path.join(path, f"*.{ext}"))
        )
        if not files:
            sys.exit(f"No image files found in directory: {path}")
        if max_frames:
            files = files[:max_frames]
        frames = np.stack([_to_gray(iio.imread(f)) for f in files])
        return frames, len(files), None

    ext = os.path.splitext(path)[1].lower()

    # --- TIFF stack ----------------------------------------------------------
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = arr[None]
        if max_frames:
            arr = arr[:max_frames]
        frames = np.stack([_to_gray(a) for a in arr])
        return frames, len(frames), None

    # --- AVI / MP4 via imageio-ffmpeg ---------------------------------------
    import imageio.v3 as iio
    try:
        meta = iio.immeta(path, plugin="pyav")
        fps = meta.get("fps")
    except Exception:
        pass
    frames = []
    for i, frame in enumerate(iio.imiter(path, plugin="pyav")):
        if max_frames and i >= max_frames:
            break
        frames.append(_to_gray(frame))
    if not frames:
        sys.exit(f"Could not read any frames from: {path}")
    return np.stack(frames), len(frames), fps


def _to_gray(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)  # RGB(A) -> luminance-ish
    return a.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Inspect a Brownian-motion video.")
    p.add_argument("path", help="video file (.avi/.mp4/.tif) or directory of frames")
    p.add_argument("--out", default="analysis/inspect_out", help="output dir for PNGs")
    p.add_argument("--samples", type=int, default=6, help="# evenly-spaced sample frames")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames read (memory)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)

    frames, n, fps = load_frames(args.path, args.max_frames)
    frames = np.asarray(frames)
    H, W = frames.shape[1:3]

    print("=" * 60)
    print(f"file        : {args.path}")
    print(f"n_frames    : {n}")
    print(f"resolution  : {W} x {H}  (W x H)")
    print(f"dtype/range : {frames.dtype}  min={frames.min():.1f} max={frames.max():.1f}")
    print(f"fps (meta)  : {fps if fps else 'unknown — set manually from uEye Cockpit'}")
    print("=" * 60)

    # sample frames
    idx = np.linspace(0, n - 1, args.samples).astype(int)
    for k, i in enumerate(idx):
        plt.imsave(os.path.join(args.out, f"frame_{i:04d}.png"), frames[i], cmap="gray")
    print(f"[saved] {args.samples} sample frames -> {args.out}/frame_*.png")

    # projections
    mx = frames.max(axis=0)
    mn = frames.min(axis=0)
    sd = frames.std(axis=0)
    for name, img in (("max", mx), ("min", mn), ("std", sd)):
        plt.imsave(os.path.join(args.out, f"proj_{name}.png"), img, cmap="gray")
    print(f"[saved] projections -> {args.out}/proj_max|min|std.png")
    print("        proj_max  = trajectory paint  (does the walk look Brownian?)")
    print("        proj_std  = moving pixels      (how many particles? static dirt?)")

    # crude drift/defocus indicator
    mean_per_frame = frames.reshape(n, -1).mean(axis=1)
    plt.figure(figsize=(7, 3))
    plt.plot(mean_per_frame)
    plt.xlabel("frame"); plt.ylabel("mean intensity")
    plt.title("Per-frame mean intensity (watch for drift in focus / illumination)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "mean_intensity.png"), dpi=110)
    print(f"[saved] mean-intensity trace -> {args.out}/mean_intensity.png")
    print("\nNext: open proj_max.png + a frame, read off the particle diameter in px,")
    print("      and tell me px count + whether you see one clean particle.")


if __name__ == "__main__":
    main()
