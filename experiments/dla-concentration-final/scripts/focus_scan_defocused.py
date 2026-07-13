#!/usr/bin/env python3
"""Focus scan over the DEFOCUSED week-4 runs (0.30 / 0.45 / 0.56 %) + the
FOCUSED 0.15 % anchor as baseline.

Question: does any time window in each clip resolve the branch structure well
enough for box-counting?  The deposit is static after grounding, so the scan
covers the WHOLE clip (a late refocus would be usable).

Per sampled frame, two independent sharpness measures:

  * grid_lapvar -- variance of the Laplacian over the static mm-grid strip
    (left edge of frame).  Content is constant, so this tracks CAMERA focus
    over time with no growth confound.
  * edge_sigma  -- effective Gaussian blur sigma of the deposit's outer edge,
    from the step-edge relation  g_peak = A / (sigma * sqrt(2*pi))  for a step
    of amplitude A blurred by a Gaussian of width sigma.  A is the median
    darkening step across the boundary; g_peak is a high quantile of the
    gradient magnitude on the boundary.  This is the number that matters:
    branches narrower than ~2*sigma are unresolvable.

Outputs: data/focus_scan.csv, figures/focus_scan.png, and the sharpest +
grounded frames as PNGs in the scratch dir for visual inspection.
"""

import csv, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"; FIGS = ROOT / "figures"
VDIR = Path(os.environ.get("WEEK4_VIDEO_DIR",
                           r"C:\dev\brownian-motion\experiments\week4-dla-no-shlomo"))
SCRATCH = Path(os.environ.get("FOCUS_SCAN_SCRATCH", tempfile.gettempdir())) / "focus_scan"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
SAMPLE_FPS = 0.5
REF_N = 12
HOLE_DARK = 40   # week-4 value: interior darkening threshold (gray levels)

RUNS = [
    dict(conc=0.30, vid="run 3 0.3.mov",               seed=(430, 330), t_gnd=138),
    dict(conc=0.45, vid="run 2 0.45 concen.mov",       seed=(327, 367), t_gnd=148),
    dict(conc=0.56, vid="run 1 0.56 Concertation.mov", seed=(463, 367), t_gnd=198),
    dict(conc=0.15, vid="run4_0.15.mov",               seed=(510, 359), t_gnd=245),  # focused anchor
]


def extract(path, out, fps):
    out.mkdir(parents=True, exist_ok=True)
    if not list(out.glob("f_*.png")):
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                        "-vf", f"fps={fps}", str(out / "f_%05d.png")], check=True)
    return sorted(out.glob("f_*.png"))


def grid_lapvar(gray, strip_w=120):
    """Laplacian variance over the static grid-paper strip (camera focus proxy)."""
    strip = gray[:, :strip_w]
    return float(cv2.Laplacian(strip, cv2.CV_32F).var())


def blob_mask(gray, ref, seed):
    """Coarse deposit blob: absolute darkening vs reference (robust to blur)."""
    offset = np.median(gray) - np.median(ref)
    darkened = ref - (gray - offset)
    m = (darkened > HOLE_DARK).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None, darkened
    # component containing (or nearest to) the seed
    sx, sy = seed
    li = lab[sy, sx]
    if li == 0:
        d = [np.hypot(st[i, 0] + st[i, 2] / 2 - sx, st[i, 1] + st[i, 3] / 2 - sy)
             if st[i, cv2.CC_STAT_AREA] > 200 else np.inf for i in range(n)]
        d[0] = np.inf
        li = int(np.argmin(d))
        if not np.isfinite(d[li]):
            return None, darkened
    return (lab == li).astype(np.uint8), darkened


def edge_sigma(gray, mask, darkened):
    """Effective Gaussian blur sigma of the deposit's outer edge.

    For an ideal step of amplitude A blurred by a Gaussian(sigma), the peak
    gradient is A / (sigma*sqrt(2pi)).  A = median interior darkening minus
    median exterior darkening (bands 4-10 px in/outside the boundary);
    g_peak = 90th percentile of |grad| on the boundary ring (the profile
    maximum is ON the boundary for a symmetric blur)."""
    if mask is None or mask.sum() < 500:
        return np.nan, np.nan, np.nan
    k = np.ones((3, 3), np.uint8)
    er4, er10 = cv2.erode(mask, k, iterations=4), cv2.erode(mask, k, iterations=10)
    di4, di10 = cv2.dilate(mask, k, iterations=4), cv2.dilate(mask, k, iterations=10)
    inner = (er4 > 0) & (er10 == 0)
    outer = (di10 > 0) & (di4 == 0)
    ring = (cv2.dilate(mask, k) > 0) & (cv2.erode(mask, k) == 0)
    if inner.sum() < 100 or outer.sum() < 100 or ring.sum() < 100:
        return np.nan, np.nan, np.nan
    A = float(np.median(darkened[inner]) - np.median(darkened[outer]))
    if A <= 5:
        return np.nan, A, np.nan
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0   # true d/dx units
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.hypot(gx, gy)
    gpk = float(np.percentile(g[ring], 90))
    if gpk <= 0:
        return np.nan, A, gpk
    return A / (gpk * np.sqrt(2 * np.pi)), A, gpk


def main():
    DATA.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    rows = []
    per_run = {}
    for run in RUNS:
        vp = VDIR / run["vid"]
        fdir = SCRATCH / f"c{run['conc']:.2f}"
        frames = extract(vp, fdir, SAMPLE_FPS)
        refs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                for p in frames[:max(1, int(REF_N * SAMPLE_FPS / 2))]]
        # reference: median of frames from the first ~6 s (pre-growth)
        ref = np.median(np.stack(refs[:6] if len(refs) >= 6 else refs), axis=0)
        rr = []
        for i, p in enumerate(frames):
            t = i / SAMPLE_FPS
            img = cv2.imread(str(p))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lv = grid_lapvar(gray)
            mask, darkened = blob_mask(gray, ref, run["seed"])
            sig, A, gpk = edge_sigma(gray, mask, darkened)
            rr.append((t, lv, sig, A, gpk, int(mask.sum()) if mask is not None else 0))
            rows.append((run["conc"], t, lv, sig, A, gpk,
                         int(mask.sum()) if mask is not None else 0, str(p.name)))
        per_run[run["conc"]] = (np.array([(a, b, c, d, e, f) for a, b, c, d, e, f in rr],
                                         dtype=float), frames, run)
        arr = per_run[run["conc"]][0]
        ok = np.isfinite(arr[:, 2]) & (arr[:, 5] > 3000)
        if ok.any():
            best = np.nanargmin(np.where(ok, arr[:, 2], np.nan))
            print(f"c={run['conc']:.2f}: sigma range "
                  f"[{np.nanmin(arr[ok, 2]):.2f}, {np.nanmax(arr[ok, 2]):.2f}] px, "
                  f"best at t={arr[best, 0]:.0f}s (sigma={arr[best, 2]:.2f}), "
                  f"grid lapvar range [{arr[:, 1].min():.1f}, {arr[:, 1].max():.1f}]",
                  flush=True)

    with open(DATA / "focus_scan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["conc", "t_s", "grid_lapvar", "edge_sigma_px", "step_A", "g_peak",
                    "blob_px", "frame"])
        w.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for conc, (arr, frames, run) in sorted(per_run.items()):
        lab = f"{conc:.2f}%" + (" (focused anchor)" if conc == 0.15 else "")
        axes[0].plot(arr[:, 0], arr[:, 1], ".-", ms=3, label=lab)
        ok = np.isfinite(arr[:, 2]) & (arr[:, 5] > 3000)
        axes[1].plot(arr[ok, 0], arr[ok, 2], ".-", ms=3, label=lab)
        axes[0].axvline(run["t_gnd"], color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel("grid-strip Laplacian variance\n(camera focus proxy)")
    axes[1].set_ylabel("deposit edge blur sigma [px]")
    axes[1].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_title("Focus scan: defocused runs vs 0.15% anchor "
                      "(dotted = grounding time)")
    fig.tight_layout()
    fig.savefig(FIGS / "focus_scan.png", dpi=130)
    print(f"-> {FIGS / 'focus_scan.png'}")

    # save the sharpest usable frame + the grounded frame per run for inspection
    for conc, (arr, frames, run) in per_run.items():
        ok = np.isfinite(arr[:, 2]) & (arr[:, 5] > 3000)
        if not ok.any():
            continue
        best = int(np.nanargmin(np.where(ok, arr[:, 2], np.nan)))
        gnd = int(np.argmin(np.abs(arr[:, 0] - run["t_gnd"])))
        for tag, idx in [("best", best), ("grounded", gnd)]:
            src = frames[idx]
            dst = SCRATCH / f"inspect_c{conc:.2f}_{tag}_t{arr[idx,0]:.0f}s.png"
            shutil.copy(src, dst)
            print(f"   {dst}")


if __name__ == "__main__":
    main()
