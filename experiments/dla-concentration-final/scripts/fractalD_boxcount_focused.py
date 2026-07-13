#!/usr/bin/env python3
"""Reliable-bucket fractal dimension: box-counting on the FOCUSED runs only
(0.02 / 0.04 / 0.06 / 0.15 %), where the deposit structure is optically resolved.

Box-counting is the correct estimator here (unlike sandbox it needs no centre,
so 0.06's off-seed problem disappears). For each run:
  * FAITHFUL mask -> largest connected component (the aggregate).
  * box-count N(s) with grid-OFFSET AVERAGING (reduces grid-placement bias).
  * branch width w (distance transform) = physical lower cutoff.
  * fit D over the principled window [w, R/3]  (above branch width, below
    finite-size), plus the median local slope over the same window as a
    robust cross-check. sigma = std of the local slope in the window.

No per-run tuning: the SAME window rule and estimator for every run.
Needs WEEK4/5_VIDEO_DIR + ffmpeg.
"""

import csv, importlib.util, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; EXP = ROOT.parent
FIGS = ROOT / "figures"; DATA = ROOT / "data"
W4 = EXP / "week4-dla-concentration" / "version 2"
W5 = EXP / "week5-dla-concentration" / "version 2"
VDIR4 = Path(os.environ.get("WEEK4_VIDEO_DIR", r"C:\dev\brownian-motion\experiments\week4-dla-no-shlomo"))
VDIR5 = Path(os.environ.get("WEEK5_VIDEO_DIR", r"C:\dev\brownian-motion\experiments\week5-dla-concentration\raw-videos"))


def load_er(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
ER4 = load_er(W4 / "scripts" / "enclosing_radius.py", "er_w4")
ER5 = load_er(W5 / "scripts" / "enclosing_radius.py", "er_w5")

RUNS = [
    dict(conc=0.02, er=ER5, wk=W5, vdir=VDIR5, tag="run1_c0.02", vid="run1_0.02con.mov"),
    dict(conc=0.04, er=ER5, wk=W5, vdir=VDIR5, tag="run2_c0.04", vid="run 2 conc 0.04.mov"),
    dict(conc=0.06, er=ER5, wk=W5, vdir=VDIR5, tag="run3_c0.06", vid="run3_0.06C.mov"),
    dict(conc=0.15, er=ER4, wk=W4, vdir=VDIR4, tag="run4_c0.15", vid="run4_0.15.mov"),
]


def meta(run):
    ppm = None
    for line in open(run["wk"] / "data" / f"radius_{run['tag']}.csv"):
        if line.startswith("# px_per_mm"):
            ppm = float(line.split("=")[1].split("+/-")[0])
    r = list(csv.DictReader([l for l in open(run["wk"] / "data" / f"radius_{run['tag']}.csv")
                             if not l.startswith("#")]))
    t_ok = [float(x["t_s"]) for x in r if int(float(x["edge"])) == 0 and float(x["M_px"]) > 0]
    return max(t_ok), ppm


def occ_mask(er, img):
    fn = getattr(er, "occluder_mask", None) or getattr(er, "wire_mask")
    return fn(img)


def faithful_mask(er, bgr, ref):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    score = 1.0 - gray / (er._flatfield_bg(gray) + 1e-6)
    darkened = ref - (gray - (np.median(gray) - np.median(ref)))
    m = (er._hysteresis(score, er.HYST_HI, er.HYST_LO).astype(np.uint8)
         & (darkened > er.CHANGE_THR)).astype(np.uint8)
    occ = (occ_mask(er, bgr) & (darkened <= er.HOLE_DARK)).astype(np.uint8)
    m[occ > 0] = 0; m[er.blue_grid_mask(bgr) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        sel = lab == i
        if st[i, cv2.CC_STAT_AREA] >= er.MIN_SIZE and score[sel].max() >= er.STRONG_CORE:
            out[sel] = 1
    return out


def largest_cc(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8)


def box_count(binary):
    """N(s) with grid-offset averaging (mean over origin shifts 0 and s//2)."""
    ys, xs = np.nonzero(binary)
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H, W = crop.shape
    sizes = np.unique(np.round(2 ** np.arange(0.5, np.log2(min(H, W) / 2), 0.1)).astype(int))
    N = []
    for s in sizes:
        cs = []
        for oy in (0, s // 2):
            for ox in (0, s // 2):
                sub = crop[oy:, ox:]
                Hc, Wc = sub.shape[0] - sub.shape[0] % s, sub.shape[1] - sub.shape[1] % s
                if Hc < s or Wc < s:
                    continue
                cs.append(int(sub[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3)).sum()))
        N.append(np.mean(cs))
    N = np.array(N); ok = N > 2
    return sizes[ok], N[ok]


def local_slope(x, y, half=2):
    lx, ly = np.log(x), np.log(y); out = np.full(len(x), np.nan)
    for i in range(half, len(x) - half):
        out[i] = np.polyfit(lx[i - half:i + half + 1], ly[i - half:i + half + 1], 1)[0]
    return out


def branch_width_px(mask):
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ridge = (d >= cv2.dilate(d, np.ones((3, 3))) - 1e-6) & (mask > 0) & (d > 1)
    return float(2 * np.median(d[ridge])) if ridge.sum() > 10 else np.nan


def grab(run):
    tmp = Path(tempfile.mkdtemp(prefix="bcf_")); rd = tmp / "ref"; rd.mkdir()
    er = run["er"]; path = run["vdir"] / run["vid"]
    t, ppm = meta(run)
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-frames:v", str(er.REF_N), str(rd / "r_%03d.png")], check=True)
    ref = np.median(np.stack([cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                              for q in sorted(rd.glob("r_*.png"))]), axis=0)
    fp = tmp / "f.png"
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
                    "-i", str(path), "-frames:v", "1", str(fp)], check=True)
    img = cv2.imread(str(fp)); shutil.rmtree(tmp, ignore_errors=True)
    return img, ref, ppm


def process(run):
    img, ref, ppm = grab(run)
    mask = largest_cc(faithful_mask(run["er"], img, ref))
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    R = np.percentile(np.hypot(xs - cx, ys - cy), 99)
    w = branch_width_px(mask)
    s, N = box_count(mask)
    ls = -local_slope(s, N)
    lo, hi = w, R / 3
    win = (s >= lo) & (s <= hi)
    D_ols = -np.polyfit(np.log(s[win]), np.log(N[win]), 1)[0] if win.sum() >= 4 else np.nan
    D_med = float(np.nanmedian(ls[win])) if win.sum() >= 4 else np.nan
    sig = float(np.nanstd(ls[win])) if win.sum() >= 4 else np.nan
    dec = np.log10(hi / lo) if (np.isfinite(w) and hi > lo) else np.nan
    # window-stability sweep: raise the lower cutoff above the branch width and
    # vary the upper cutoff; a genuine fractal D is stable, a shoulder-biased one
    # falls as the lower cutoff climbs past the branch width.
    def fitw(a, b):
        m = (s >= a) & (s <= b)
        return (-np.polyfit(np.log(s[m]), np.log(N[m]), 1)[0]) if m.sum() >= 4 else np.nan
    sweep = {f"{k:g}w": {f"R/{u}": fitw(k * w, R / u) for u in (4, 3, 2)}
             for k in (1, 1.5, 2, 2.5)}
    return dict(conc=run["conc"], ppm=ppm, R_mm=R / ppm, w_mm=w / ppm, dec=dec,
                D_ols=D_ols, D_med=D_med, sig=sig, npts=int(win.sum()),
                s_mm=s / ppm, ls=ls, lo=lo / ppm, hi=hi / ppm, wmm=w / ppm,
                mask=mask, sweep=sweep)


def pipeline_D():
    d = {}
    for wk in (W4, W5):
        for r in csv.DictReader(open(wk / "data" / "fractalD_summary.csv")):
            d[round(float(r["conc"]), 2)] = float(r["D_boxcount"])
    return d


def main():
    old = pipeline_D()
    rows = [process(r) for r in RUNS]
    print(f"\n{'conc':>5} {'R[mm]':>6} {'w[mm]':>6} {'win[dec]':>8} {'#pts':>5} "
          f"{'D_old':>6} {'D_OLS':>6} {'D_med':>6} {'sigma':>6} {'reliable':>9}")
    for r in rows:
        rel = "YES" if (r["dec"] >= 0.5 and r["sig"] < 0.15 and r["npts"] >= 5) else "check"
        print(f"{r['conc']:5.2f} {r['R_mm']:6.1f} {r['w_mm']:6.2f} {r['dec']:8.2f} "
              f"{r['npts']:5d} {old[r['conc']]:6.3f} {r['D_ols']:6.3f} {r['D_med']:6.3f} "
              f"{r['sig']:6.3f} {rel:>9}")

    print("\n=== window-stability sweep: D over [k*w, R/u] "
          "(lower cutoff climbing past the branch width) ===")
    for r in rows:
        print(f"\n {r['conc']:.2f}%  (branch width w = {r['w_mm']:.2f} mm, R = {r['R_mm']:.1f} mm)")
        print(f"   {'':6}" + "".join(f"{u:>8}" for u in ("R/4", "R/3", "R/2")))
        for k, d in r["sweep"].items():
            print(f"   {k:6}" + "".join(
                f"{d[f'R/{u}']:8.3f}" if np.isfinite(d[f'R/{u}']) else f"{'-':>8}"
                for u in (4, 3, 2)))

    # local-slope grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 10)); axes = axes.ravel()
    for ax, r in zip(axes, rows):
        ax.semilogx(r["s_mm"], r["ls"], "o-", ms=4, color="C0")
        ax.axvspan(r["lo"], r["hi"], color="C2", alpha=0.13, label="fit window [w, R/3]")
        ax.axvline(r["wmm"], color="C3", ls="--", lw=1.3, label=f"branch width {r['wmm']:.2f}mm")
        ax.axhline(1.71, color="k", ls="--", lw=1, label="DLA 1.71")
        ax.axhline(2.0, color="gray", ls=":", lw=1)
        ax.axhline(r["D_ols"], color="C1", lw=1.6, label=f"D = {r['D_ols']:.2f}")
        ax.set_ylim(1.0, 2.15); ax.grid(alpha=0.3)
        ax.set_title(f"{r['conc']:.2f}%  D={r['D_ols']:.3f}±{r['sig']:.2f} "
                     f"(med {r['D_med']:.2f})", fontsize=12)
        ax.set_xlabel("box size [mm]"); ax.set_ylabel("local D (box-count)")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Reliable bucket — box-counting local dimension (focused runs)", fontsize=14)
    fig.tight_layout(); fig.savefig(FIGS / "fractalD_boxcount_focused.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"\n-> {FIGS/'fractalD_boxcount_focused.png'}")
    return rows


if __name__ == "__main__":
    main()
