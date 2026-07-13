#!/usr/bin/env python3
"""Boundary (perimeter) fractal dimension -- the fallback quantity for 0.30 %.

The 0.30 % mass-D is window-limited (coarse compact branches + sigma ~2.5 px
blur leave <0.7 decade).  The DEPOSIT OUTLINE, however, is resolved at scales
above ~3 sigma, so the boundary's box-counting dimension D_b is measurable.
D_b is a DIFFERENT quantity from the mass dimension (for a compact object
with a fractal coast, mass D -> 2 while 1 < D_b < 2) and is reported as such.

Boundary = mask minus its erosion (1-px outline of the faithful seed-CC
mask, no RL -- deconvolution ringing must not touch a boundary measurement).
Window [3*sigma_blur, R/3]; same offset-averaged box counter; same
window-stability sweep; validated on the focused 0.15 % anchor and the sharp
0.56 % run (their D_b under synthetic blur to 0.30's sigma must not move).
"""

import csv, importlib.util, os, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "figures"; DATA = ROOT / "data"

# reuse the deblur module's helpers (masking, box count, sigma estimator)
spec = importlib.util.spec_from_file_location("fdd", HERE / "fractalD_deblur.py")
fdd = importlib.util.module_from_spec(spec)
sys.modules["fdd"] = fdd
spec.loader.exec_module.__self__ if False else spec.loader.exec_module(fdd)

ARMS = [
    dict(tag="c0.15_clean", conc=0.15, vid="run4_0.15.mov",               seed=(510, 359), ppm=48.60, t0=244, t1=244, rl=False),
    dict(tag="c0.15_blur2.5", conc=0.15, vid="run4_0.15.mov",             seed=(510, 359), ppm=48.60, t0=244, t1=244, rl=False, blur_to=2.52),
    dict(tag="c0.56_gnd",   conc=0.56, vid="run 1 0.56 Concertation.mov", seed=(463, 367), ppm=49.53, t0=198, t1=198, rl=False),
    dict(tag="c0.56_blur2.5", conc=0.56, vid="run 1 0.56 Concertation.mov", seed=(463, 367), ppm=49.53, t0=198, t1=198, rl=False, blur_to=2.52),
    dict(tag="c0.45_gnd",   conc=0.45, vid="run 2 0.45 concen.mov",       seed=(327, 367), ppm=48.16, t0=148, t1=148, rl=False),
    dict(tag="c0.30_gnd",   conc=0.30, vid="run 3 0.3.mov",               seed=(430, 330), ppm=47.75, t0=138, t1=138, rl=False),
    dict(tag="c0.30_late",  conc=0.30, vid="run 3 0.3.mov",               seed=(430, 330), ppm=47.75, t0=300, t1=344, rl=False),
]


def perimeter_D(arm):
    refs = {}
    r = fdd.process(arm, refs)          # builds the faithful seed-CC mask
    mask = r["mask"]
    k = np.ones((3, 3), np.uint8)
    bnd = (mask - cv2.erode(mask, k)).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    R = np.percentile(np.hypot(xs - cx, ys - cy), 99)
    sig = r["sig_pre"]
    lo = max(3 * sig, 3.0) if np.isfinite(sig) else 3.0
    hi = R / 3
    s, N = fdd.box_count(bnd)
    ls = -fdd.local_slope(s, N)
    win = (s >= lo) & (s <= hi)
    D = -np.polyfit(np.log(s[win]), np.log(N[win]), 1)[0] if win.sum() >= 4 else np.nan
    Dm = float(np.nanmedian(ls[win])) if win.sum() >= 4 else np.nan
    sg = float(np.nanstd(ls[win])) if win.sum() >= 4 else np.nan
    dec = np.log10(hi / lo) if hi > lo > 0 else np.nan

    def fitw(a, b):
        m = (s >= a) & (s <= b)
        return (-np.polyfit(np.log(s[m]), np.log(N[m]), 1)[0]) if m.sum() >= 4 else np.nan
    sweep = {f"{kk:g}x": {f"R/{u}": fitw(kk * lo, R / u) for u in (4, 3, 2)}
             for kk in (1, 1.5, 2, 2.5)}
    return dict(arm=arm, bnd=bnd, sig=sig, R=R, lo=lo, hi=hi, s=s, N=N, ls=ls,
                D=D, Dm=Dm, sg=sg, dec=dec, npts=int(win.sum()), sweep=sweep)


def main():
    FIGS.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    rows = [perimeter_D(a) for a in ARMS]
    print(f"\n{'tag':>14} {'sig':>5} {'lo':>5} {'R':>5} {'dec':>5} {'#':>3} "
          f"{'D_b':>6} {'med':>6} {'std':>6}")
    for r in rows:
        print(f"{r['arm']['tag']:>14} {r['sig']:5.2f} {r['lo']:5.1f} {r['R']:5.0f} "
              f"{r['dec']:5.2f} {r['npts']:3d} {r['D']:6.3f} {r['Dm']:6.3f} {r['sg']:6.3f}")
    print("\n=== window-stability sweep: D_b over [k*lo, R/u] ===")
    for r in rows:
        print(f"\n {r['arm']['tag']}  (lo = {r['lo']:.1f} px, R = {r['R']:.0f} px)")
        print(f"   {'':6}" + "".join(f"{u:>8}" for u in ("R/4", "R/3", "R/2")))
        for kk, d in r["sweep"].items():
            print(f"   {kk:6}" + "".join(
                f"{d[f'R/{u}']:8.3f}" if np.isfinite(d[f'R/{u}']) else f"{'-':>8}"
                for u in (4, 3, 2)))

    with open(DATA / "fractalD_perimeter.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "conc", "sigma_px", "lo_px", "R_px", "decades", "npts",
                    "Db_ols", "Db_med", "std_localslope"])
        for r in rows:
            w.writerow([r["arm"]["tag"], r["arm"]["conc"], f"{r['sig']:.3f}",
                        f"{r['lo']:.2f}", f"{r['R']:.1f}", f"{r['dec']:.3f}",
                        r["npts"], f"{r['D']:.4f}", f"{r['Dm']:.4f}", f"{r['sg']:.4f}"])

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, r in enumerate(rows):
        ax = axes[0, i] if i < 4 else axes[1, i - 4]
        ax.imshow(r["bnd"], cmap="binary", vmin=0, vmax=1)
        ax.set_title(f"{r['arm']['tag']}  D_b={r['D']:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1, 3]
    for r in rows:
        ax.semilogx(r["s"] / r["arm"]["ppm"], r["ls"], ".-", ms=3,
                    label=f"{r['arm']['tag']} ({r['D']:.2f})")
    ax.set_ylim(0.9, 1.8); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_xlabel("box size [mm]"); ax.set_ylabel("local D_b")
    for j in range(len(rows), 7):
        (axes[0, j] if j < 4 else axes[1, j - 4]).axis("off")
    fig.suptitle("Boundary (perimeter) box-counting dimension -- distinct from mass D", fontsize=13)
    fig.tight_layout()
    out = FIGS / "fractalD_perimeter.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
