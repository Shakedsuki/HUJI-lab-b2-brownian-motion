#!/usr/bin/env python3
"""Corrected fractal-dimension extraction, demonstrated on ONE run (0.15 %).

The fix, step by step:
  1. FAITHFUL mask (no hole-fill / no absolute-darkening) -- true branch geometry.
  2. Measure the branch width w (distance-transform ridge) -- the physical lower
     cutoff below which box-counting just measures 'solid'.
  3. Box-count over the FRACTAL window [w, R/2.5] (above the branch width, below
     finite-size) instead of the old [8px, R/8] (which sat below w).
  4. Report the local slope's flatness over that window; if it is not flat over
     >= 0.5 decade, the object has no fractal regime -> declare compact.
  5. Also measure the BOUNDARY (perimeter) fractal dimension, which is
     well-defined for a compact-but-rough deposit and is concentration-sensitive.

Compares old vs corrected. Needs WEEK4_VIDEO_DIR + ffmpeg.
"""

import csv, importlib.util, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W4 = ROOT.parent / "week4-dla-concentration" / "version 2"
FIGS = ROOT / "figures"
TAG, CONC, VIDEO = "run4_c0.15", 0.15, "run4_0.15.mov"


def load_er(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
er = load_er(W4 / "scripts" / "enclosing_radius.py", "er_w4")


def meta():
    ppm = seed = None
    for line in open(W4 / "data" / f"radius_{TAG}.csv"):
        if line.startswith("#"):
            if "px_per_mm" in line: ppm = float(line.split("=")[1].split("+/-")[0])
            if "seed" in line: seed = eval(line.split("=", 1)[1])
    r = list(csv.DictReader([l for l in open(W4 / "data" / f"radius_{TAG}.csv") if not l.startswith("#")]))
    t_ok = [float(x["t_s"]) for x in r if int(float(x["edge"])) == 0 and float(x["M_px"]) > 0]
    return max(t_ok), seed, ppm


def faithful_mask(bgr, ref, hi=er.HYST_HI, lo=er.HYST_LO):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    score = 1.0 - gray / (er._flatfield_bg(gray) + 1e-6)
    darkened = ref - (gray - (np.median(gray) - np.median(ref)))
    m = (er._hysteresis(score, hi, lo).astype(np.uint8) & (darkened > er.CHANGE_THR)).astype(np.uint8)
    occ = (er.occluder_mask(bgr) & (darkened <= er.HOLE_DARK)).astype(np.uint8)
    m[occ > 0] = 0; m[er.blue_grid_mask(bgr) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        sel = lab == i
        if st[i, cv2.CC_STAT_AREA] >= er.MIN_SIZE and score[sel].max() >= er.STRONG_CORE:
            out[sel] = 1
    return out


def disc_gate(mask, seed, min_px=30):
    ys, xs = np.nonzero(mask)
    R = np.percentile(np.hypot(xs - seed[0], ys - seed[1]), 99)
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_px:
            keep[i] = np.hypot(cent[i, 0] - seed[0], cent[i, 1] - seed[1]) <= 1.15 * R
    return keep[lab].astype(np.uint8)


def box_count(binary):
    ys, xs = np.nonzero(binary)
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sizes = np.unique(np.round(2 ** np.arange(0.5, np.log2(min(crop.shape) / 2), 0.1)).astype(int))
    N, H, W = [], *crop.shape
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        N.append(int(crop[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3)).sum()))
    N = np.array(N); ok = N > 2
    return sizes[ok], N[ok]


def sandbox(mask, seed):
    """M(<r): deposit pixels within radius r of the seed."""
    ys, xs = np.nonzero(mask)
    r = np.hypot(xs - seed[0], ys - seed[1])
    R = np.percentile(r, 99)
    edges = np.geomspace(4, R, 40)
    M = np.array([(r <= e).sum() for e in edges])
    ok = M > 0
    return edges[ok], M[ok]


def local_slope(x, y, half=2):
    lx, ly = np.log(x), np.log(y); out = np.full(len(x), np.nan)
    for i in range(half, len(x) - half):
        out[i] = np.polyfit(lx[i - half:i + half + 1], ly[i - half:i + half + 1], 1)[0]
    return out


def fit(x, y, lo, hi, sign=-1):
    w = (x >= lo) & (x <= hi)
    if w.sum() < 4: return np.nan, 0, np.nan
    D = sign * np.polyfit(np.log(x[w]), np.log(y[w]), 1)[0]
    ls = sign * local_slope(x, y)
    return D, int(w.sum()), float(np.nanstd(ls[w]))


def branch_width_px(mask):
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    mx = cv2.dilate(d, np.ones((3, 3)))
    ridge = (d >= mx - 1e-6) & (mask > 0) & (d > 1)
    return float(2 * np.median(d[ridge]))


def main():
    t, seed, ppm = meta()
    tmp = Path(tempfile.mkdtemp(prefix="fdx_"))
    rd = tmp / "ref"; rd.mkdir()
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-i",
                    str(er.VIDEO_DIR / VIDEO), "-frames:v", str(er.REF_N),
                    str(rd / "r_%03d.png")], check=True)
    ref = np.median(np.stack([cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                              for q in sorted(rd.glob("r_*.png"))]), axis=0)
    fp = tmp / "f.png"
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
                    "-i", str(er.VIDEO_DIR / VIDEO), "-frames:v", "1", str(fp)], check=True)
    img = cv2.imread(str(fp)); shutil.rmtree(tmp, ignore_errors=True)

    mask = disc_gate(faithful_mask(img, ref), seed)
    ys, xs = np.nonzero(mask)
    R = np.percentile(np.hypot(xs - seed[0], ys - seed[1]), 99)
    w = branch_width_px(mask)
    bound = (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))).astype(np.uint8)

    s_bx, N_bx = box_count(mask)
    r_sb, M_sb = sandbox(mask, seed)
    s_bd, N_bd = box_count(bound)

    # OLD window vs CORRECTED fractal window
    D_old, n_old, _ = fit(s_bx, N_bx, 8, R / 8)
    lo_c, hi_c = w, R / 2.5                       # above branch width, below finite-size
    D_box, n_box, flat_box = fit(s_bx, N_bx, lo_c, hi_c)
    D_sand, n_sand, flat_sand = fit(r_sb, M_sb, lo_c, hi_c, sign=+1)
    D_bnd, n_bnd, flat_bnd = fit(s_bd, N_bd, 8, R / 3)

    dec_c = np.log10(hi_c / lo_c)
    print(f"\n=== 0.15 % corrected fractal-D extraction "
          f"(faithful mask, {ppm:.1f} px/mm, R={R:.0f}px={R/ppm:.1f}mm) ===")
    print(f"branch width w = {w:.0f} px = {w/ppm:.2f} mm")
    print(f"corrected fractal window [w, R/2.5] = [{lo_c/ppm:.2f}, {hi_c/ppm:.2f}] mm "
          f"= {dec_c:.2f} decades\n")
    print(f"{'estimator':28} {'D':>6} {'#pts':>5} {'slope std in window':>20}")
    print(f"{'OLD box-count [8px,R/8]':28} {D_old:6.3f} {n_old:5d} {'(below branch width)':>20}")
    print(f"{'box-count [w, R/2.5]':28} {D_box:6.3f} {n_box:5d} {flat_box:20.3f}")
    print(f"{'sandbox M(<r) [w, R/2.5]':28} {D_sand:6.3f} {n_sand:5d} {flat_sand:20.3f}")
    print(f"{'boundary/perimeter [8px,R/3]':28} {D_bnd:6.3f} {n_bnd:5d} {flat_bnd:20.3f}")
    flat = "FLAT (real D)" if flat_box < 0.10 else "NOT flat -> no fractal window (compact)"
    print(f"\nmass-D scaling region verdict: {flat}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].semilogx(s_bx / ppm, -local_slope(s_bx, N_bx), "o-", ms=4, color="C0",
                   label="box-count local D")
    ax[0].semilogx(r_sb / ppm, local_slope(r_sb, M_sb), "s-", ms=4, color="C1",
                   label="sandbox local D")
    ax[0].axvspan(lo_c / ppm, hi_c / ppm, color="C2", alpha=0.12,
                  label="corrected fractal window")
    ax[0].axvline(w / ppm, color="C3", ls="--", lw=1.5, label=f"branch width {w/ppm:.2f} mm")
    ax[0].axvline(R / 8 / ppm, color="gray", ls=":", lw=1.5, label="old window top R/8")
    ax[0].axhline(1.71, color="k", ls="--", lw=1); ax[0].axhline(2.0, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("scale [mm]"); ax[0].set_ylabel("local mass dimension")
    ax[0].set_ylim(1.0, 2.2); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
    ax[0].set_title("mass dimension: old window sat below the branch width")

    ax[1].semilogx(s_bd / ppm, -local_slope(s_bd, N_bd), "o-", ms=4, color="C4",
                   label="boundary local D")
    ax[1].axhline(1.0, color="k", ls=":", lw=1); ax[1].axhline(2.0, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("scale [mm]"); ax[1].set_ylabel("local boundary dimension")
    ax[1].set_ylim(0.9, 2.1); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)
    ax[1].set_title(f"boundary (perimeter) dimension = {D_bnd:.3f}")
    fig.suptitle("0.15 % anchor — corrected fractal-dimension extraction", fontsize=14)
    fig.tight_layout()
    out = FIGS / "fractalD_extract_c0.15.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
