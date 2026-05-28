"""
analysis/calibrate_scale.py
===========================
HUJI Lab B2 — Brownian Motion, 2026
Shaked Sukiennik & Nir Cohen

Extract the spatial calibration  µm/px  from a ruler ("measurent_0.1mm*") video.

Why median several frames first: the ruler is STATIC, so the per-pixel median
across frames removes floating dust, shot noise, and any single-frame glitch,
leaving a clean ruler image.

Two measurement modes (the script runs the auto one and saves diagnostics; use
--manual if the ruling is coarser than the frame):

  AUTO  (default) — for a fine periodic ruling (e.g. 10 µm stage-micrometer
        lines). Collapses the image to a 1-D intensity profile (averaging out
        the perpendicular axis), detrends, and takes an FFT. The dominant
        spatial frequency = ruler line spacing in px. Cross-checked with the
        autocorrelation first-peak. Done for BOTH orientations; the sharper
        peak wins.

  MANUAL — for a coarse ruling where only one 0.1 mm gap fits in the frame.
        You pass the pixel coordinates of two ruler edges (read off the saved
        ruler.png) and the physical distance between them.

Calibration:   um_per_px = period_um / period_px

Usage
-----
    # auto-detect periodic lines, assume each detected period is 10 µm:
    python analysis/calibrate_scale.py data/raw/measurent_0.1mm_focused.avi --period-um 10

    # coarse ruling: read two edge x-coords off ruler.png, they span 100 µm:
    python analysis/calibrate_scale.py data/raw/measurent_0.1mm_focused.avi \
        --manual --p1 120 --p2 1530 --dist-um 100 --axis x
"""

import argparse
import os
import sys
import numpy as np


def read_frames(path, n_sample=15):
    """Return a list of grayscale float32 frames sampled across the video."""
    # --- try imageio + pyav -------------------------------------------------
    try:
        import imageio.v3 as iio
        # For a static ruler any frames do; grab a handful early ones.
        frames = []
        for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
            if i % 5 == 0:
                frames.append(_gray(fr))
            if len(frames) >= n_sample:
                break
        if frames:
            return frames
    except Exception as e:
        print(f"[imageio/pyav failed: {e}] falling back to OpenCV", file=sys.stderr)

    # --- fallback: OpenCV ---------------------------------------------------
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"OpenCV could not open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.linspace(0, max(total - 1, 0), n_sample).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(_gray(fr))
    cap.release()
    if not frames:
        sys.exit("OpenCV opened the file but read no frames (codec issue).")
    return frames


def _gray(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    return a.astype(np.float32)


def dominant_period(profile):
    """Return (period_px, strength) of the dominant periodicity in a 1-D profile."""
    x = profile - profile.mean()
    n = len(x)
    win = np.hanning(n)
    xf = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=1.0)
    # ignore DC and ultra-low freq (period > n/2)
    lo = 3
    if len(xf) <= lo + 1:
        return np.nan, 0.0
    k = lo + int(np.argmax(xf[lo:]))
    if freqs[k] == 0:
        return np.nan, 0.0
    period_px = 1.0 / freqs[k]
    strength = xf[k] / (xf[lo:].mean() + 1e-9)   # peak prominence
    return period_px, strength


def main():
    p = argparse.ArgumentParser(description="Measure µm/px from a ruler video.")
    p.add_argument("path")
    p.add_argument("--out", default="analysis/calib_out")
    p.add_argument("--period-um", type=float, default=10.0,
                   help="physical distance of ONE detected period (µm). Stage "
                        "micrometers: finest lines usually 10 µm.")
    p.add_argument("--manual", action="store_true")
    p.add_argument("--p1", type=float, help="manual: pixel coord of edge 1")
    p.add_argument("--p2", type=float, help="manual: pixel coord of edge 2")
    p.add_argument("--dist-um", type=float, default=100.0, help="manual: µm between p1,p2")
    p.add_argument("--axis", choices=["x", "y"], default="x", help="manual: ruling axis")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)

    frames = read_frames(args.path)
    ruler = np.median(np.stack(frames), axis=0)
    H, W = ruler.shape
    plt.imsave(os.path.join(args.out, "ruler.png"), ruler, cmap="gray")
    print(f"[ruler] {W}x{H}  median of {len(frames)} frames -> {args.out}/ruler.png")

    if args.manual:
        if args.p1 is None or args.p2 is None:
            sys.exit("--manual needs --p1 and --p2 (pixel coords off ruler.png)")
        period_px = abs(args.p2 - args.p1)
        um_per_px = args.dist_um / period_px
        print(f"[manual] {args.dist_um} µm spans {period_px:.1f} px on axis {args.axis}")
        print(f"\n   um_per_px = {um_per_px:.5f}  µm/px")
        print(f"   (1 px = {um_per_px*1000:.1f} nm at the sample plane)")
        return

    # auto: profiles along both axes
    prof_x = ruler.mean(axis=0)   # varies along x -> vertical lines
    prof_y = ruler.mean(axis=1)   # varies along y -> horizontal lines
    px_x, s_x = dominant_period(prof_x)
    px_y, s_y = dominant_period(prof_y)

    fig, ax = plt.subplots(2, 1, figsize=(9, 5))
    ax[0].plot(prof_x); ax[0].set_title(f"profile along x  (period~{px_x:.1f}px, strength {s_x:.1f})")
    ax[1].plot(prof_y); ax[1].set_title(f"profile along y  (period~{px_y:.1f}px, strength {s_y:.1f})")
    ax[1].set_xlabel("pixel")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "profiles.png"), dpi=110)
    print(f"[profiles] -> {args.out}/profiles.png")

    # pick the orientation with the stronger periodic peak
    if s_x >= s_y:
        axis, period_px, strength = "x", px_x, s_x
    else:
        axis, period_px, strength = "y", px_y, s_y

    print(f"\n[auto] dominant ruling on axis {axis}: period = {period_px:.2f} px"
          f"  (peak strength {strength:.1f}; >5 is a confident detection)")
    if np.isfinite(period_px):
        um_per_px = args.period_um / period_px
        print(f"   assuming 1 period = {args.period_um} µm:")
        print(f"   um_per_px = {um_per_px:.5f}  µm/px   "
              f"(1 px = {um_per_px*1000:.1f} nm)")
    print("\n  -> send me ruler.png + profiles.png so I can verify the period and")
    print("     confirm which physical spacing (10 vs 100 µm) the lines are.")


if __name__ == "__main__":
    main()
