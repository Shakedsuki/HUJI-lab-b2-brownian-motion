#!/usr/bin/env python3
"""Fractal dimension for the week-4 'defocused' runs (0.30 / 0.45 / 0.56 %).

The focus scan (focus_scan_defocused.py) showed the label 'defocused' is only
partly true: the deposit-edge blur sigma is 0.5-0.6 px for 0.56 % (sharper
than the focused 0.15 % anchor, 0.8 px), ~1.6 px for 0.45 %, and 2-3 px for
0.30 %.  Branch structure is visibly resolved in all three at the right
frame.  So we box-count exactly as in the reliable bucket, with one extra
honesty rule and two validations:

  * the fit window's LOWER cutoff is  max(branch width, 3*sigma_blur)  --
    below ~3 sigma the mask boundary is optics, not structure;
  * the 0.15 % anchor is run through this exact pipeline (expect ~1.87);
  * the anchor frame is synthetically Gaussian-blurred to sigma 1.5 and
    3.0 px (the 0.45 / 0.30 blur levels) and re-measured: the D shift under
    known blur bounds the blur-induced bias for the real runs.

Estimator identical to fractalD_boxcount_focused.py: faithful mask (no
hole-fill), largest / seed component, box-counting with grid-offset
averaging, OLS + median local slope over the window, window-stability sweep.
Needs WEEK4_VIDEO_DIR + ffmpeg.
"""

import csv, importlib.util, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "figures"; DATA = ROOT / "data"
W4 = ROOT.parent / "week4-dla-concentration" / "version 2"
VDIR = Path(os.environ.get("WEEK4_VIDEO_DIR",
                           r"C:\dev\brownian-motion\experiments\week4-dla-no-shlomo"))


def load_er(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


er = load_er(W4 / "scripts" / "enclosing_radius.py", "er_w4")

# t_frame: grounded frame (largest static structure) unless a sharper usable
# window exists; 0.30 gets a mid-growth cross-check frame (sharper, smaller).
RUNS = [
    dict(conc=0.30, vid="run 3 0.3.mov",               seed=(430, 330), t=138, ppm=47.75, tag="c0.30"),
    dict(conc=0.30, vid="run 3 0.3.mov",               seed=(430, 330), t=60,  ppm=47.75, tag="c0.30_mid"),
    dict(conc=0.45, vid="run 2 0.45 concen.mov",       seed=(327, 367), t=148, ppm=48.16, tag="c0.45"),
    dict(conc=0.56, vid="run 1 0.56 Concertation.mov", seed=(463, 367), t=198, ppm=49.53, tag="c0.56"),
    dict(conc=0.15, vid="run4_0.15.mov",               seed=(510, 359), t=244, ppm=48.60, tag="c0.15_anchor"),
    dict(conc=0.15, vid="run4_0.15.mov",               seed=(510, 359), t=244, ppm=48.60, tag="c0.15_blur1.5", blur=1.5),
    dict(conc=0.15, vid="run4_0.15.mov",               seed=(510, 359), t=244, ppm=48.60, tag="c0.15_blur3.0", blur=3.0),
]


def grab(run):
    tmp = Path(tempfile.mkdtemp(prefix="fdd_")); rd = tmp / "ref"; rd.mkdir()
    path = VDIR / run["vid"]
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-frames:v", str(er.REF_N), str(rd / "r_%03d.png")], check=True)
    ref = np.median(np.stack([cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                              for q in sorted(rd.glob("r_*.png"))]), axis=0)
    fp = tmp / "f.png"
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{run['t']:.3f}",
                    "-i", str(path), "-frames:v", "1", str(fp)], check=True)
    img = cv2.imread(str(fp)); shutil.rmtree(tmp, ignore_errors=True)
    if run.get("blur"):
        img = cv2.GaussianBlur(img, (0, 0), run["blur"])
    return img, ref


def faithful_mask(bgr, ref):
    """Same as the reliable-bucket mask (fractalD_boxcount_focused.py)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    score = 1.0 - gray / (er._flatfield_bg(gray) + 1e-6)
    darkened = ref - (gray - (np.median(gray) - np.median(ref)))
    m = (er._hysteresis(score, er.HYST_HI, er.HYST_LO).astype(np.uint8)
         & (darkened > er.CHANGE_THR)).astype(np.uint8)
    occ = (er.occluder_mask(bgr) & (darkened <= er.HOLE_DARK)).astype(np.uint8)
    m[occ > 0] = 0; m[er.blue_grid_mask(bgr) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        sel = lab == i
        if st[i, cv2.CC_STAT_AREA] >= er.MIN_SIZE and score[sel].max() >= er.STRONG_CORE:
            out[sel] = 1
    return out


def seed_cc(mask, seed):
    """Component at/nearest the seed (fallback: largest)."""
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    li = lab[seed[1], seed[0]]
    if li == 0:
        big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 500]
        if not big:
            li = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        else:
            li = min(big, key=lambda i: np.hypot(cent[i, 0] - seed[0], cent[i, 1] - seed[1]))
    return (lab == li).astype(np.uint8)


def edge_sigma(gray, mask, darkened):
    """Blur sigma of the outer edge: g_peak = A / (sigma*sqrt(2pi)) for a
    Gaussian-blurred step (same estimator as the focus scan)."""
    k = np.ones((3, 3), np.uint8)
    inner = (cv2.erode(mask, k, iterations=4) > 0) & (cv2.erode(mask, k, iterations=10) == 0)
    outer = (cv2.dilate(mask, k, iterations=10) > 0) & (cv2.dilate(mask, k, iterations=4) == 0)
    ring = (cv2.dilate(mask, k) > 0) & (cv2.erode(mask, k) == 0)
    A = float(np.median(darkened[inner]) - np.median(darkened[outer]))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gpk = float(np.percentile(np.hypot(gx, gy)[ring], 90))
    return A / (gpk * np.sqrt(2 * np.pi)) if gpk > 0 and A > 5 else np.nan


def box_count(binary):
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


def process(run):
    img, ref = grab(run)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    darkened = ref - (gray - (np.median(gray) - np.median(ref)))
    mask = seed_cc(faithful_mask(img, ref), run["seed"])
    sig = edge_sigma(gray, mask, darkened)
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    R = np.percentile(np.hypot(xs - cx, ys - cy), 99)
    w = branch_width_px(mask)
    lo = max(w, 3 * sig) if np.isfinite(sig) else w
    hi = R / 3
    s, N = box_count(mask)
    ls = -local_slope(s, N)
    win = (s >= lo) & (s <= hi)
    D_ols = -np.polyfit(np.log(s[win]), np.log(N[win]), 1)[0] if win.sum() >= 4 else np.nan
    D_med = float(np.nanmedian(ls[win])) if win.sum() >= 4 else np.nan
    sg = float(np.nanstd(ls[win])) if win.sum() >= 4 else np.nan
    dec = np.log10(hi / lo) if hi > lo > 0 else np.nan

    def fitw(a, b):
        m = (s >= a) & (s <= b)
        return (-np.polyfit(np.log(s[m]), np.log(N[m]), 1)[0]) if m.sum() >= 4 else np.nan
    sweep = {f"{k:g}x": {f"R/{u}": fitw(k * lo, R / u) for u in (4, 3, 2)}
             for k in (1, 1.5, 2, 2.5)}
    return dict(run=run, img=img, mask=mask, sig=sig, R=R, w=w, lo=lo, hi=hi,
                s=s, N=N, ls=ls, win=win, D_ols=D_ols, D_med=D_med, sg=sg,
                dec=dec, npts=int(win.sum()), sweep=sweep)


def main():
    FIGS.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    rows = [process(r) for r in RUNS]

    print(f"\n{'tag':>15} {'t[s]':>5} {'sig[px]':>7} {'w[px]':>6} {'lo[px]':>6} "
          f"{'R[px]':>6} {'dec':>5} {'#pts':>5} {'D_OLS':>6} {'D_med':>6} {'std':>6}")
    for r in rows:
        print(f"{r['run']['tag']:>15} {r['run']['t']:5.0f} {r['sig']:7.2f} {r['w']:6.1f} "
              f"{r['lo']:6.1f} {r['R']:6.0f} {r['dec']:5.2f} {r['npts']:5d} "
              f"{r['D_ols']:6.3f} {r['D_med']:6.3f} {r['sg']:6.3f}")

    print("\n=== window-stability sweep: D over [k*lo, R/u] ===")
    for r in rows:
        print(f"\n {r['run']['tag']}  (lo = {r['lo']:.1f} px, R = {r['R']:.0f} px)")
        print(f"   {'':6}" + "".join(f"{u:>8}" for u in ("R/4", "R/3", "R/2")))
        for k, d in r["sweep"].items():
            print(f"   {k:6}" + "".join(
                f"{d[f'R/{u}']:8.3f}" if np.isfinite(d[f'R/{u}']) else f"{'-':>8}"
                for u in (4, 3, 2)))

    with open(DATA / "fractalD_defocused.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["tag", "conc", "t_s", "sigma_px", "w_px", "lo_px", "R_px",
                       "decades", "npts", "D_ols", "D_med", "std_localslope"])
        for r in rows:
            wcsv.writerow([r["run"]["tag"], r["run"]["conc"], r["run"]["t"],
                           f"{r['sig']:.3f}", f"{r['w']:.2f}", f"{r['lo']:.2f}",
                           f"{r['R']:.1f}", f"{r['dec']:.3f}", r["npts"],
                           f"{r['D_ols']:.4f}", f"{r['D_med']:.4f}", f"{r['sg']:.4f}"])

    # per-run diagnostic figure: frame / mask / local slope
    for r in rows:
        ppm = r["run"]["ppm"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"{r['run']['tag']}  t={r['run']['t']}s  "
                          f"sigma={r['sig']:.2f}px")
        axes[1].imshow(r["mask"], cmap="binary", vmin=0, vmax=1)
        axes[1].set_title(f"faithful mask (seed CC), w={r['w']:.1f}px")
        for a in axes[:2]:
            a.set_xticks([]); a.set_yticks([])
        ax = axes[2]
        ax.semilogx(r["s"] / ppm, r["ls"], "o-", ms=4, color="C0")
        ax.axvspan(r["lo"] / ppm, r["hi"] / ppm, color="C2", alpha=0.13,
                   label="fit window [max(w,3sig), R/3]")
        ax.axhline(r["D_ols"], color="C1", lw=1.6, label=f"D = {r['D_ols']:.3f}")
        ax.axhline(1.71, color="k", ls="--", lw=1, label="DLA 1.71")
        ax.axhline(2.0, color="gray", ls=":", lw=1)
        ax.set_ylim(1.0, 2.15); ax.grid(alpha=0.3)
        ax.set_xlabel("box size [mm]"); ax.set_ylabel("local D")
        ax.set_title(f"D = {r['D_ols']:.3f} +/- {r['sg']:.2f} (med {r['D_med']:.2f}, "
                     f"{r['dec']:.2f} dec)")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        out = FIGS / f"fractalD_defocused_{r['run']['tag']}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
