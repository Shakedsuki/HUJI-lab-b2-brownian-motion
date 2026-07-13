#!/usr/bin/env python3
"""Pre/post probe for the fractal-dimension mask fix, on ONE reliable run.

Target: 0.15 % (week-4 run4, run4_0.15.mov) -- the focused, lamp-less anchor,
the only dense run whose ramified structure is optically resolved, so the
faithful (hole-fill-removed) mask is trustworthy and any pre/post gap is
attributable purely to the segmentation artefact.

  * FILLED   = er.deposit_mask()          -- current pipeline (absolute-darkening
                                             pathway + interior hole-filling)
  * FAITHFUL = local-contrast hysteresis only, same exclusions & seed gate,
                                             NO absolute-darkening, NO hole-fill

Box-counting D over the quoted window [8 px, R/8] for both, plus the full local
slope d(logN)/d(logs). Saves a side-by-side mask+diff+slope figure.
Needs the raw video (WEEK4_VIDEO_DIR) and ffmpeg.
"""

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT.parent
W4 = EXP / "week4-dla-concentration" / "version 2"
FIGS = ROOT / "figures"

TAG = "run4_c0.15"
CONC = 0.15


def load_er(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


er = load_er(W4 / "scripts" / "enclosing_radius.py", "er_w4")


def run_meta():
    ppm = seed = None
    ts, edge, M = [], [], []
    for line in open(W4 / "data" / f"radius_{TAG}.csv"):
        if line.startswith("#"):
            if "px_per_mm" in line:
                ppm = float(line.split("=")[1].split("+/-")[0])
            if "seed" in line:
                seed = eval(line.split("=", 1)[1])
            continue
    r = list(csv.DictReader([l for l in open(W4 / "data" / f"radius_{TAG}.csv")
                             if not l.startswith("#")]))
    for x in r:
        ts.append(float(x["t_s"])); edge.append(int(float(x["edge"])))
        M.append(float(x["M_px"]))
    t_ok = [t for t, e, m in zip(ts, edge, M) if e == 0 and m > 0]
    return max(t_ok), seed, ppm


def faithful_mask(bgr, ref_gray, hi=er.HYST_HI, lo=er.HYST_LO):
    """deposit_mask() with the two compactifying steps removed: no
    `| (darkened > HOLE_DARK)` absolute-darkening term, no interior hole-fill.
    Everything else (flat-field score, change gate, occluder/grid exclusion,
    strong-core component gate) is identical."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = er._flatfield_bg(gray)
    score = 1.0 - gray / (bg + 1e-6)
    offset = np.median(gray) - np.median(ref_gray)
    darkened = ref_gray - (gray - offset)
    m = er._hysteresis(score, hi, lo).astype(np.uint8)          # <-- no abs-dark
    changed = darkened > er.CHANGE_THR
    m = (m & changed).astype(np.uint8)
    occ = er.occluder_mask(bgr)
    occ = (occ & (darkened <= er.HOLE_DARK)).astype(np.uint8)
    grid_all = er.blue_grid_mask(bgr)
    m[occ > 0] = 0
    m[grid_all > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < er.MIN_SIZE:
            continue
        sel = lab == i
        if score[sel].max() >= er.STRONG_CORE:
            out[sel] = 1
    return out                                                  # <-- no hole-fill


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


def box_counts(mask):
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    smax = min(crop.shape) // 2
    sizes = np.unique(np.round(2 ** np.arange(0.5, np.log2(smax), 0.125)).astype(int))
    N, H, W = [], *crop.shape
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        blocks = crop[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3))
        N.append(int(blocks.sum()))
    N = np.array(N)
    ok = N > 4
    return sizes[ok], N[ok]


def fit_D(sizes, N, lo, hi):
    win = (sizes >= lo) & (sizes <= hi)
    if win.sum() < 4:
        return np.nan, 0
    D = -np.polyfit(np.log(sizes[win]), np.log(N[win]), 1)[0]
    return D, int(win.sum())


def local_slope(x, y, half=2):
    lx, ly = np.log(x), np.log(y)
    out = np.full(len(x), np.nan)
    for i in range(half, len(x) - half):
        out[i] = np.polyfit(lx[i - half:i + half + 1], ly[i - half:i + half + 1], 1)[0]
    return out


def analyse(mask, seed):
    ys, xs = np.nonzero(mask)
    R = np.percentile(np.hypot(xs - seed[0], ys - seed[1]), 99)
    sizes, N = box_counts(mask)
    D, npts = fit_D(sizes, N, 8, R / 8)
    dec = np.log10((R / 8) / 8)
    return dict(R=R, sizes=sizes, N=N, D=D, npts=npts, decades=dec,
                area=int(mask.sum()))


def main():
    t_last, seed, ppm = run_meta()
    tmp = Path(tempfile.mkdtemp(prefix="maskfix_"))
    try:
        path = er.VIDEO_DIR / "run4_0.15.mov"
        refdir = tmp / "ref"; refdir.mkdir()
        subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-i",
                        str(path), "-frames:v", str(er.REF_N),
                        str(refdir / "r_%03d.png")], check=True)
        ref = np.median(np.stack([
            cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for q in sorted(refdir.glob("r_*.png"))]), axis=0)
        fp = tmp / "f.png"
        subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error",
                        "-ss", f"{t_last:.3f}", "-i", str(path), "-frames:v", "1",
                        str(fp)], check=True)
        img = cv2.imread(str(fp))
    finally:
        pass

    filled = disc_gate(er.deposit_mask(img, ref), seed)
    faith = disc_gate(faithful_mask(img, ref), seed)
    shutil.rmtree(tmp, ignore_errors=True)

    a_fill = analyse(filled, seed)
    a_faith = analyse(faith, seed)

    print(f"\n=== 0.15 % (run4) box-counting D, filled vs faithful mask "
          f"(t={t_last:.0f}s, {ppm:.1f} px/mm) ===")
    print(f"{'':10} {'D[8,R/8]':>9} {'#pts':>5} {'decades':>8} {'area px':>9} "
          f"{'fill %':>7}")
    fillpct = 100 * (a_fill['area'] - a_faith['area']) / a_faith['area']
    print(f"{'FILLED':10} {a_fill['D']:9.3f} {a_fill['npts']:5d} "
          f"{a_fill['decades']:8.2f} {a_fill['area']:9d} {'':>7}")
    print(f"{'FAITHFUL':10} {a_faith['D']:9.3f} {a_faith['npts']:5d} "
          f"{a_faith['decades']:8.2f} {a_faith['area']:9d} {fillpct:6.1f}%")
    print(f"\nDelta D (filled - faithful) = {a_fill['D'] - a_faith['D']:+.3f}")
    print(f"quoted summary D_boxcount for 0.15 = 1.886")

    # ---- figure ----
    onlyfill = (filled > 0) & (faith == 0)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.28, wspace=0.15)
    for ax, (m, ttl) in zip(
            [fig.add_subplot(gs[0, i]) for i in range(3)],
            [(faith, f"FAITHFUL mask\nD = {a_faith['D']:.3f}"),
             (filled, f"FILLED mask (current)\nD = {a_fill['D']:.3f}"),
             (None, f"filled-only pixels\n(+{fillpct:.0f}% area)")]):
        if m is None:
            rgb = np.zeros((*faith.shape, 3), np.uint8)
            rgb[faith > 0] = (90, 90, 90)
            rgb[onlyfill] = (220, 40, 40)
            ax.imshow(rgb)
        else:
            ax.imshow(m, cmap="binary", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=13); ax.set_xticks([]); ax.set_yticks([])

    axl = fig.add_subplot(gs[1, :])
    for a, lab, col in [(a_faith, "faithful", "C0"), (a_fill, "filled", "C3")]:
        ls = -local_slope(a["sizes"], a["N"])
        axl.semilogx(a["sizes"] / ppm, ls, "o-", ms=4, color=col,
                     label=f"{lab}  (D[8,R/8] = {a['D']:.3f})")
    axl.axhline(1.71, color="k", ls="--", lw=1, label="2D DLA 1.71")
    axl.axhline(2.0, color="gray", ls=":", lw=1, label="compact 2.0")
    axl.axvline((a_fill["R"] / 8) / ppm, color="C2", ls="--", lw=1,
                label="R/8 (window top)")
    axl.set_xlabel("box size [mm]"); axl.set_ylabel("local box-counting D")
    axl.set_ylim(1.0, 2.15); axl.grid(alpha=0.3); axl.legend(fontsize=10, ncol=2)
    axl.set_title("local slope d(logN)/d(logs): filled vs faithful")
    fig.suptitle("0.15 % anchor — fractal-dimension mask fix probe", fontsize=15)
    out = FIGS / "maskfix_probe_c0.15.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
