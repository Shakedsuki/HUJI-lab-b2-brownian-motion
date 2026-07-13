#!/usr/bin/env python3
"""Corrected fractal-dimension extraction for ALL seven runs.

For each run, on the grounded frame (t = last edge-free time):
  1. FAITHFUL mask (no hole-fill / no absolute-darkening).
  2. branch width w (distance-transform ridge).
  3. sandbox M(<r) from the seed; fit D over the FRACTAL window [w, R/2.5]
     (above the branch width, below finite-size) -- the corrected mass D,
     with a data-driven uncertainty = std of the local slope in the window.
  4. boundary (perimeter) dimension as a secondary descriptor.
Compares against the published (filled-mask) D_boxcount.

Outputs: console table, data/fractalD_corrected.csv, two figures.
Needs WEEK4_VIDEO_DIR / WEEK5_VIDEO_DIR + ffmpeg.
"""

import csv, importlib.util, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT.parent
FIGS = ROOT / "figures"; DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
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
    dict(conc=0.30, er=ER4, wk=W4, vdir=VDIR4, tag="run3_c0.30", vid="run 3 0.3.mov"),
    dict(conc=0.45, er=ER4, wk=W4, vdir=VDIR4, tag="run2_c0.45", vid="run 2 0.45 concen.mov"),
    dict(conc=0.56, er=ER4, wk=W4, vdir=VDIR4, tag="run1_c0.56", vid="run 1 0.56 Concertation.mov"),
]

# focus quality (week-4 README): only run4/0.15 is fully focused; the other
# week-4 runs are defocused -> their faithful masks capture fringe, unreliable.
FOCUSED = {0.02: True, 0.04: True, 0.06: True, 0.15: True,
           0.30: False, 0.45: False, 0.56: False}


def meta(run):
    ppm = seed = None
    for line in open(run["wk"] / "data" / f"radius_{run['tag']}.csv"):
        if line.startswith("#"):
            if "px_per_mm" in line: ppm = float(line.split("=")[1].split("+/-")[0])
            if line.startswith("# seed"): seed = eval(line.split("=", 1)[1])
    r = list(csv.DictReader([l for l in open(run["wk"] / "data" / f"radius_{run['tag']}.csv")
                             if not l.startswith("#")]))
    t_ok = [float(x["t_s"]) for x in r if int(float(x["edge"])) == 0 and float(x["M_px"]) > 0]
    return max(t_ok), seed, ppm


def occ_mask(er, img):
    fn = getattr(er, "occluder_mask", None) or getattr(er, "wire_mask")
    return fn(img)


def faithful_mask(er, bgr, ref, hi=None, lo=None):
    hi = er.HYST_HI if hi is None else hi
    lo = er.HYST_LO if lo is None else lo
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    score = 1.0 - gray / (er._flatfield_bg(gray) + 1e-6)
    darkened = ref - (gray - (np.median(gray) - np.median(ref)))
    m = (er._hysteresis(score, hi, lo).astype(np.uint8) & (darkened > er.CHANGE_THR)).astype(np.uint8)
    occ = (occ_mask(er, bgr) & (darkened <= er.HOLE_DARK)).astype(np.uint8)
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
    ys, xs = np.nonzero(mask)
    r = np.hypot(xs - seed[0], ys - seed[1]); R = np.percentile(r, 99)
    edges = np.geomspace(4, R, 40)
    M = np.array([(r <= e).sum() for e in edges]); ok = M > 0
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
    return D, int(w.sum()), float(np.nanstd(sign * local_slope(x, y)[w]))


def branch_width_px(mask):
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ridge = (d >= cv2.dilate(d, np.ones((3, 3))) - 1e-6) & (mask > 0) & (d > 1)
    return float(2 * np.median(d[ridge])) if ridge.sum() > 10 else np.nan


def pipeline_D():
    d = {}
    for wk in (W4, W5):
        for r in csv.DictReader(open(wk / "data" / "fractalD_summary.csv")):
            d[round(float(r["conc"]), 2)] = float(r["D_boxcount"])
    return d


def process(run):
    t, seed, ppm = meta(run)
    tmp = Path(tempfile.mkdtemp(prefix="fdxall_")); rd = tmp / "ref"; rd.mkdir()
    er = run["er"]; path = run["vdir"] / run["vid"]
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-frames:v", str(er.REF_N), str(rd / "r_%03d.png")], check=True)
    ref = np.median(np.stack([cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                              for q in sorted(rd.glob("r_*.png"))]), axis=0)
    fp = tmp / "f.png"
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
                    "-i", str(path), "-frames:v", "1", str(fp)], check=True)
    img = cv2.imread(str(fp)); shutil.rmtree(tmp, ignore_errors=True)

    mask = disc_gate(faithful_mask(er, img, ref), seed)
    ys, xs = np.nonzero(mask)
    R = np.percentile(np.hypot(xs - seed[0], ys - seed[1]), 99)
    w = branch_width_px(mask)
    r_sb, M_sb = sandbox(mask, seed)
    bound = (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))).astype(np.uint8)
    s_bd, N_bd = box_count(bound)

    lo_c, hi_c = w, R / 2.5
    D_sand, n_sand, sig = fit(r_sb, M_sb, lo_c, hi_c, sign=+1)
    D_bnd, _, _ = fit(s_bd, N_bd, 8, R / 3)
    dec = np.log10(hi_c / lo_c) if (np.isfinite(w) and hi_c > lo_c) else np.nan
    return dict(conc=run["conc"], ppm=ppm, R_mm=R / ppm, w_mm=w / ppm, dec=dec,
                D_corr=D_sand, sig=sig, D_bnd=D_bnd, focused=FOCUSED[run["conc"]],
                r_sb=r_sb / ppm, ls_sb=local_slope(r_sb, M_sb), lo=lo_c / ppm, hi=hi_c / ppm,
                wmm=w / ppm)


def main():
    old = pipeline_D()
    rows = [process(r) for r in RUNS]

    print(f"\n{'conc':>5} {'R[mm]':>6} {'w[mm]':>6} {'win[dec]':>8} "
          f"{'D_old':>6} {'D_corr':>7} {'sigma':>6} {'D_bnd':>6} {'focus':>6} {'reliable':>9}")
    for r in rows:
        rel = "yes" if (r["focused"] and r["dec"] > 0.55 and r["sig"] < 0.12) else "marginal"
        print(f"{r['conc']:5.2f} {r['R_mm']:6.1f} {r['w_mm']:6.2f} {r['dec']:8.2f} "
              f"{old[r['conc']]:6.3f} {r['D_corr']:7.3f} {r['sig']:6.3f} {r['D_bnd']:6.3f} "
              f"{'Y' if r['focused'] else 'n':>6} {rel:>9}")

    with open(DATA / "fractalD_corrected.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["conc", "D_old_filled", "D_corrected_sandbox", "sigma",
                     "D_boundary", "R_mm", "branch_w_mm", "window_decades", "focused"])
        for r in rows:
            wr.writerow([r["conc"], f"{old[r['conc']]:.3f}", f"{r['D_corr']:.3f}",
                         f"{r['sig']:.3f}", f"{r['D_bnd']:.3f}", f"{r['R_mm']:.1f}",
                         f"{r['w_mm']:.2f}", f"{r['dec']:.2f}", int(r["focused"])])

    # ---- figure 1: D vs conc, old vs corrected ----
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    cs = [r["conc"] for r in rows]
    ax.plot(cs, [old[c] for c in cs], "s--", ms=8, color="C3", alpha=0.7,
            label="published (filled mask, 8px-R/8)")
    for r in rows:
        mk = "o" if r["focused"] else "x"
        ax.errorbar(r["conc"], r["D_corr"], yerr=r["sig"], fmt=mk, ms=9, capsize=4,
                    color="C0", mfc=("C0" if r["focused"] else "white"))
    ax.plot([], [], "o", color="C0", label="corrected (faithful, sandbox > w) — focused")
    ax.plot([], [], "x", color="C0", label="corrected — defocused (unreliable)")
    ax.axhline(1.71, color="k", ls="--", lw=1.4, label="2D DLA theory: 1.71")
    for r in rows:
        ax.annotate(f"{r['D_corr']:.2f}", (r["conc"], r["D_corr"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=9, color="C0")
    ax.set_xlabel("CuSO$_4$ concentration [%]"); ax.set_ylabel("fractal dimension  D")
    ax.set_ylim(1.35, 2.02); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "D_vs_concentration_corrected.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: sandbox local-slope grid (reliability) ----
    fig, axes = plt.subplots(3, 3, figsize=(15, 12)); axes = axes.ravel()
    for ax, r in zip(axes, rows):
        ax.semilogx(r["r_sb"], r["ls_sb"], "o-", ms=3, color="C1")
        ax.axvspan(r["lo"], r["hi"], color="C2", alpha=0.12)
        ax.axvline(r["wmm"], color="C3", ls="--", lw=1.2)
        ax.axhline(1.71, color="k", ls="--", lw=1); ax.axhline(2.0, color="gray", ls=":", lw=1)
        ax.axhline(r["D_corr"], color="C0", lw=1.5)
        ax.set_ylim(1.0, 2.2); ax.grid(alpha=0.3)
        ax.set_title(f"{r['conc']:.2f}%  D={r['D_corr']:.2f}±{r['sig']:.2f}  "
                     f"{'focused' if r['focused'] else 'DEFOCUS'}", fontsize=11)
        ax.set_xlabel("r [mm]"); ax.set_ylabel("local D (sandbox)")
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle("Sandbox local dimension per run (green = fit window above branch width)",
                 fontsize=14)
    fig.tight_layout(); fig.savefig(FIGS / "fractalD_corrected_grid.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"\n-> {DATA/'fractalD_corrected.csv'}")
    print(f"-> {FIGS/'D_vs_concentration_corrected.png'}")
    print(f"-> {FIGS/'fractalD_corrected_grid.png'}")


if __name__ == "__main__":
    main()
