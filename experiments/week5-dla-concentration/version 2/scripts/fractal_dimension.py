#!/usr/bin/env python3
"""Fractal dimension per CuSO4 concentration (week 5, deliverable 3).

For each run the deposit mask is extracted from LATE video frames (the largest
aggregate that still fits in the frame: the last sampled time with edge == 0 in
data/radius_<tag>.csv, plus two earlier check frames), using the same
segmentation as enclosing_radius.py (temporal change + flat-field hysteresis +
wire/blue-grid exclusion + seed gate).  Two independent estimators per frame:

  * box counting          N(s) ~ s^-D,   8 px <= s <= R/8
  * mass-radius (seed)    M(<r) ~ r^D,   30 px <= r <= 0.8 R,
                          each annulus corrected for wire occlusion

The quoted D per concentration is the mean over {frames} x {estimators} with
an error combining that spread with a +/-0.03 segmentation-threshold
systematic (hysteresis HI varied 0.12-0.18, as in week 3).

Run AFTER enclosing_radius.py:  python scripts/fractal_dimension.py
"""

import csv
import subprocess
import tempfile
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import enclosing_radius as er

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

THRESH_SCAN = (0.12, 0.15, 0.18)   # hysteresis HI values for the systematic
FRAME_FRACS = (1.0, 0.85, 0.70)    # of the last edge-free time


def load_run_meta(run):
    """Last edge-free sample time + seed from the kinetics CSV."""
    rows, seed, ppm = [], None, None
    with open(DATA / f"radius_{run['tag']}.csv") as fh:
        for line in fh:
            if line.startswith("#"):
                if "seed" in line:
                    seed = eval(line.split("=", 1)[1])
                if "px_per_mm" in line:
                    ppm = float(line.split("=", 1)[1].split("+/-")[0])
                continue
            rows.append(line)
    r = list(csv.DictReader(rows))
    ts = [float(x["t_s"]) for x in r]
    edge = [int(float(x["edge"])) for x in r]
    M = [float(x["M_px"]) for x in r]
    t_ok = [t for t, e, m in zip(ts, edge, M) if e == 0 and m > 0]
    return max(t_ok), seed, ppm


def grab(path, t_s, tmp):
    fp = tmp / f"f_{t_s:.1f}.png"
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error",
                    "-ss", f"{t_s:.3f}", "-i", str(path), "-frames:v", "1", str(fp)],
                   check=True)
    return cv2.imread(str(fp))


def reference(path, tmp):
    refdir = tmp / "ref"
    refdir.mkdir()
    subprocess.run([er.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-frames:v", str(er.REF_N), str(refdir / "r_%03d.png")], check=True)
    refs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for p in sorted(refdir.glob("r_*.png"))]
    return np.median(np.stack(refs), axis=0)


def disc_gate(mask, seed, min_px=30):
    """Keep components inside the aggregate disc (centroid within 1.15 R99 of
    the seed) and above speckle size.  Single late frames carry no cluster
    memory, and seed-connectivity alone drops branches whose root crosses the
    static wire-shadow band; in a frame dominated by the deposit the disc gate
    is the robust criterion (same as the week-3 still-photo analysis)."""
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


# ------------------------------------------------------------- estimators ---

def box_count(binary, sizes):
    H, W = binary.shape
    out = []
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        blocks = binary[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3))
        out.append(blocks.sum())
    return np.array(out)


def boxcount_D(mask, R):
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sizes = np.unique(np.round(
        2 ** np.arange(1, np.log2(min(crop.shape) / 2), 0.25)).astype(int))
    N = box_count(crop, sizes)
    ok = N > 0
    sizes, N = sizes[ok], N[ok]
    win = (sizes >= 8) & (sizes <= R / 8)
    D, dD, b = er.fit_loglog(sizes[win], N[win])
    return -D, dD, sizes, N, win, b


def massradius_D(mask, wire, seed, R):
    cx, cy = seed
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)
    visible = wire == 0
    edges = np.geomspace(4, R, 40)
    mass, valid = [], []
    for r0, r1 in zip(edges[:-1], edges[1:]):
        ann = (r >= r0) & (r < r1)
        vis = (ann & visible).sum()
        tot = ann.sum()
        if tot == 0 or vis / tot < 0.4:
            mass.append(0.0)
            valid.append(False)
            continue
        mass.append(mask[ann].sum() * tot / vis)
        valid.append(True)
    mass = np.cumsum(mass)
    rmid = np.sqrt(edges[:-1] * edges[1:])
    valid = np.array(valid) & (mass > 0)
    win = valid & (rmid >= 30) & (rmid <= 0.8 * R)
    D, dD, b = er.fit_loglog(rmid[win], mass[win])
    return D, dD, rmid, mass, win, b


# ------------------------------------------------------------ diagnostics ---

def diagnostics(run, img, mask, wire, seed, R, bc, mr, t_s):
    Dbc, dDbc, sizes, N, winb, bb = bc
    Dmr, dDmr, rmid, mass, winm, bm = mr
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    vis[mask > 0] = (220, 30, 30)
    vis[wire > 0] = (30, 90, 220)
    axes[0].imshow(vis)
    th = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(seed[0] + R * np.cos(th), seed[1] + R * np.sin(th), "y--", lw=1)
    axes[0].set_title(f"{run['label']}%  t={t_s:.0f}s\n(red = deposit, blue = wire mask)")
    axes[0].axis("off")

    axes[1].loglog(sizes, N, "o", ms=4, color="gray")
    axes[1].loglog(sizes[winb], N[winb], "o", ms=5, color="C3", label="fit window")
    ss = np.array([sizes[winb].min(), sizes[winb].max()])
    axes[1].loglog(ss, np.exp(bb) * ss ** (-Dbc), "k-", lw=1)
    axes[1].set_xlabel("box size s (px)"); axes[1].set_ylabel("N(s)")
    axes[1].set_title(f"box counting: D = {Dbc:.3f} +/- {dDbc:.3f}")
    axes[1].legend()

    axes[2].loglog(rmid, np.where(mass > 0, mass, np.nan), "o", ms=4, color="gray")
    axes[2].loglog(rmid[winm], mass[winm], "o", ms=5, color="C0", label="fit window")
    rr = np.array([rmid[winm].min(), rmid[winm].max()])
    axes[2].loglog(rr, np.exp(bm) * rr ** Dmr, "k-", lw=1)
    axes[2].set_xlabel("r from seed (px)"); axes[2].set_ylabel("M(<r) (occl.-corr.)")
    axes[2].set_title(f"mass-radius: D = {Dmr:.3f} +/- {dDmr:.3f}")
    axes[2].legend()

    fig.tight_layout()
    out = FIGS / f"fractal_{run['tag']}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def summary_figure(rows):
    """D vs concentration, both estimators + combined."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    concs = [r["conc"] for r in rows]
    ax.errorbar(concs, [r["D"] for r in rows], yerr=[r["dD"] for r in rows],
                fmt="o", ms=6, capsize=4, color="C0", label="combined")
    ax.plot(concs, [r["Dbc"] for r in rows], "s", ms=4, color="C3", alpha=0.6,
            label="box counting")
    ax.plot(concs, [r["Dmr"] for r in rows], "^", ms=4, color="C2", alpha=0.6,
            label="mass-radius")
    ax.axhline(1.71, color="k", ls="--", lw=1, label="DLA theory (2D): 1.71")
    ax.set_xlabel("CuSO4 concentration [%]")
    ax.set_ylabel("fractal dimension D")
    ax.set_title("fractal dimension vs CuSO4 concentration")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "D_vs_concentration.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    summary = []
    for run in er.RUNS:
        t_last, seed, ppm = load_run_meta(run)
        path = er.VIDEO_DIR / run["file"]
        tmp = Path(tempfile.mkdtemp(prefix=f"w5fd_{run['tag']}_"))
        try:
            ref = reference(path, tmp)
            Ds_bc, Ds_mr = [], []
            diag = None
            for frac in FRAME_FRACS:
                t_s = frac * t_last
                img = grab(path, t_s, tmp)
                wire = er.wire_mask(img)
                for hi in THRESH_SCAN:
                    mask = disc_gate(
                        er.deposit_mask(img, ref, hi=hi, lo=er.HYST_LO), seed)
                    if mask.sum() < 500:
                        continue
                    ys, xs = np.nonzero(mask)
                    R = np.percentile(np.hypot(xs - seed[0], ys - seed[1]), 99)
                    bc = boxcount_D(mask, R)
                    mr = massradius_D(mask, wire, seed, R)
                    Ds_bc.append(bc[0])
                    Ds_mr.append(mr[0])
                    if frac == 1.0 and hi == er.HYST_HI:
                        diag = diagnostics(run, img, mask, wire, seed, R, bc, mr, t_s)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        Ds = np.array(Ds_bc + Ds_mr)
        D = Ds.mean()
        dD = float(np.sqrt(Ds.std(ddof=1) ** 2 + 0.03 ** 2))
        summary.append(dict(conc=run["conc"], D=D, dD=dD,
                            Dbc=float(np.mean(Ds_bc)), Dmr=float(np.mean(Ds_mr)),
                            n=len(Ds), t=t_last))
        print(f"{run['tag']}:  t_meas <= {t_last:.0f}s  "
              f"D_box = {np.mean(Ds_bc):.3f}  D_massradius = {np.mean(Ds_mr):.3f}  "
              f"->  D = {D:.2f} +/- {dD:.2f}   ({len(Ds)} estimates)  fig: {diag}",
              flush=True)

    with open(DATA / "fractalD_summary.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["conc", "D", "dD", "D_boxcount", "D_massradius", "n_estimates",
                     "t_measured_s"])
        for r in summary:
            wr.writerow([r["conc"], f"{r['D']:.3f}", f"{r['dD']:.3f}",
                         f"{r['Dbc']:.3f}", f"{r['Dmr']:.3f}", r["n"], f"{r['t']:.0f}"])
    summary_figure(summary)
    print(f"summary -> {DATA / 'fractalD_summary.csv'}, {FIGS / 'D_vs_concentration.png'}",
          flush=True)


if __name__ == "__main__":
    main()
