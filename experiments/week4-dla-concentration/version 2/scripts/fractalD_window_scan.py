#!/usr/bin/env python3
"""Disambiguate growth-regime vs scaling-window systematics in the measured D.

For every run of weeks 4 and 5 (week 5 = control: same pipeline read D = 1.54
at 0.02%), on the same late-frame mask used for the quoted D:

  1. local slope d(log N)/d(log s) of box counting across ALL scales -- a real
     fractal shows a plateau; its extent (in mm) is the honest scaling range;
  2. a window-stability scan: D fitted over a grid of (s_min, s_max) windows in
     PHYSICAL units -- if D drifts systematically as the window narrows to
     intermediate scales, the quoted value is crossover-contaminated;
  3. branch-width estimate w (2x the median ridge value of the distance
     transform): the meaningful lower cutoff, in mm;
  4. erosion sensitivity: D on the mask eroded by 1 px -- large shifts mean
     the measurement rides on 1-px edge dilation (segmentation/anti-aliasing);
  5. mass-radius local slope d(log M)/d(log r), correction on AND off.

Console tables + per-run local-slope figure in figures/windowscan_<tag>.png.
"""

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
W4 = HERE.parent
W5 = W4.parent.parent / "week5-dla-concentration" / "version 2"
FIGS = W4 / "figures"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ER4 = load_module(W4 / "scripts" / "enclosing_radius.py", "er_w4")
ER5 = load_module(W5 / "scripts" / "enclosing_radius.py", "er_w5")

RUNS = ([dict(er=ER4, root=W4, week=4, **r) for r in ER4.RUNS] +
        [dict(er=ER5, root=W5, week=5, **r) for r in ER5.RUNS])


def run_meta(root, tag):
    ppm = seed = None
    body = []
    for line in open(root / "data" / f"radius_{tag}.csv"):
        if line.startswith("#"):
            if "px_per_mm" in line:
                ppm = float(line.split("=")[1].split("+/-")[0])
            if "seed" in line:
                seed = eval(line.split("=", 1)[1])
            continue
        body.append(line)
    r = list(csv.DictReader(body))
    ts = [float(x["t_s"]) for x in r]
    edge = [int(float(x["edge"])) for x in r]
    M = [float(x["M_px"]) for x in r]
    t_ok = [t for t, e, m in zip(ts, edge, M) if e == 0 and m > 0]
    return max(t_ok), seed, ppm


def disc_gate(mask, seed, min_px=30):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    R = np.percentile(np.hypot(xs - seed[0], ys - seed[1]), 99)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        keep[i] = np.hypot(cent[i, 0] - seed[0], cent[i, 1] - seed[1]) <= 1.15 * R
    return keep[lab].astype(np.uint8)


def get_mask(run, t_s, seed):
    er = run["er"]
    path = er.VIDEO_DIR / run["file"]
    tmp = Path(tempfile.mkdtemp(prefix="ws_"))
    try:
        refdir = tmp / "ref"; refdir.mkdir()
        subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-i",
                        str(path), "-frames:v", str(er.REF_N),
                        str(refdir / "r_%03d.png")], check=True)
        refs = [cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                for q in sorted(refdir.glob("r_*.png"))]
        ref = np.median(np.stack(refs), axis=0)
        fp = tmp / "f.png"
        subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error",
                        "-ss", f"{t_s:.3f}", "-i", str(path), "-frames:v", "1",
                        str(fp)], check=True)
        img = cv2.imread(str(fp))
        mask = disc_gate(er.deposit_mask(img, ref), seed)
        try:
            occ = er.occluder_mask(img)
        except AttributeError:          # week 5 module: wire only
            occ = er.wire_mask(img)
        return img, mask, occ
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def box_counts(mask):
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    smax = min(crop.shape) // 2
    sizes = np.unique(np.round(2 ** np.arange(0.5, np.log2(smax), 0.125)).astype(int))
    N = []
    H, W = crop.shape
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        blocks = crop[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3))
        N.append(blocks.sum())
    N = np.array(N)
    ok = N > 4
    return sizes[ok], N[ok]


def local_slope(x, y, half=2):
    """centered log-log slope over +/-half points"""
    lx, ly = np.log(x), np.log(y)
    out = np.full(len(x), np.nan)
    for i in range(half, len(x) - half):
        out[i] = np.polyfit(lx[i - half:i + half + 1], ly[i - half:i + half + 1], 1)[0]
    return out


def fit_D(sizes, N, lo, hi):
    win = (sizes >= lo) & (sizes <= hi)
    if win.sum() < 4:
        return np.nan
    return -np.polyfit(np.log(sizes[win]), np.log(N[win]), 1)[0]


def branch_width_mm(mask, ppm):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # ridge = local maxima of the distance transform inside the mask
    mx = cv2.dilate(dist, np.ones((3, 3)))
    ridge = (dist >= mx - 1e-6) & (mask > 0) & (dist > 1)
    if ridge.sum() < 10:
        return np.nan
    return float(2 * np.median(dist[ridge]) / ppm)


def massradius_slopes(mask, occ, seed, R, ppm):
    cx, cy = seed
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    edges = np.geomspace(4, R, 60)
    rmid = np.sqrt(edges[:-1] * edges[1:])
    out = {}
    for name, visible in [("corr", occ == 0), ("nocorr", np.ones_like(mask, bool))]:
        mass = []
        for r0, r1 in zip(edges[:-1], edges[1:]):
            ann = (r >= r0) & (r < r1)
            vis = (ann & visible).sum(); tot = ann.sum()
            mass.append(mask[ann].sum() * (tot / vis if (tot and vis / tot >= 0.4) else 1.0))
        cum = np.cumsum(mass)
        ok = cum > 0
        out[name] = (rmid[ok], cum[ok])
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    print(f"{'run':14s} {'w[mm]':>6s} {'D(1px-eroded)':>13s}  window-scan D (s_min x s_max, mm)")
    for run in RUNS:
        t_last, seed, ppm = run_meta(run["root"], run["tag"])
        img, mask, occ = get_mask(run, t_last, seed)
        ys, xs = np.nonzero(mask)
        R = np.percentile(np.hypot(xs - seed[0], ys - seed[1]), 99)
        sizes, N = box_counts(mask)
        w_mm = branch_width_mm(mask, ppm)

        # window scan in physical units
        smins_mm = [0.08, 0.12, 0.16, 0.25, 0.4]
        smaxs_mm = [0.6, 1.0, 1.6, 2.5]
        grid = np.full((len(smins_mm), len(smaxs_mm)), np.nan)
        for i, lo in enumerate(smins_mm):
            for j, hi in enumerate(smaxs_mm):
                if hi / lo < 3:
                    continue
                grid[i, j] = fit_D(sizes, N, lo * ppm, hi * ppm)

        # erosion sensitivity (default window: 8px..R/8 as quoted)
        D0 = fit_D(sizes, N, 8, R / 8)
        er1 = cv2.erode(mask, np.ones((3, 3), np.uint8))
        s1, N1 = box_counts(er1) if er1.sum() > 1000 else (None, None)
        D1 = fit_D(s1, N1, 8, R / 8) if s1 is not None else np.nan

        tagw = f"w{run['week']}:{run['conc']:.2f}"
        print(f"{tagw:14s} {w_mm:6.2f} {D0:6.3f}->{D1:5.3f}  ", end="")
        print(" | ".join(
            ",".join(f"{grid[i, j]:.2f}" if np.isfinite(grid[i, j]) else "  - "
                     for j in range(len(smaxs_mm)))
            for i in range(len(smins_mm))))

        # figure: local slope + mass-radius slopes
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ls = local_slope(sizes, N)
        ax[0].semilogx(sizes / ppm, -ls, "o-", ms=3)
        ax[0].axhline(1.71, color="k", ls="--", lw=1, label="DLA 1.71")
        ax[0].axhline(2.0, color="gray", ls=":", lw=1)
        ax[0].axvline(w_mm, color="C3", ls="--", lw=1, label=f"branch width {w_mm:.2f} mm")
        ax[0].axvline(R / 8 / ppm, color="C2", ls="--", lw=1, label="R/8 (quoted window top)")
        ax[0].set_xlabel("box size [mm]"); ax[0].set_ylabel("local D (box counting)")
        ax[0].set_ylim(1.0, 2.2); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax[0].set_title(f"{tagw}: local box-counting slope")

        mr = massradius_slopes(mask, occ, seed, R, ppm)
        for name, (rr, mm_) in mr.items():
            lsr = local_slope(rr, mm_)
            ax[1].semilogx(rr / ppm, lsr, "o-", ms=3,
                           label=f"mass-radius local slope ({name})")
        ax[1].axhline(1.71, color="k", ls="--", lw=1)
        ax[1].axhline(2.0, color="gray", ls=":", lw=1)
        ax[1].set_xlabel("r from seed [mm]"); ax[1].set_ylabel("local D (mass-radius)")
        ax[1].set_ylim(1.0, 2.6); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
        ax[1].set_title("mass-radius local slope, occl. corr. on/off")
        fig.tight_layout()
        out = FIGS / f"windowscan_w{run['week']}_{run['tag']}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
    print("\nfigures -> figures/windowscan_*.png")
    print("columns: s_min in {0.08,0.12,0.16,0.25,0.4} mm (rows) x "
          "s_max in {0.6,1.0,1.6,2.5} mm (cols)")


if __name__ == "__main__":
    main()
