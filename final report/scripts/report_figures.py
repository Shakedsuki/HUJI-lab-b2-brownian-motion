#!/usr/bin/env python3
"""Combined report figures for the DLA-vs-CuSO4-concentration experiment.

Merges the two measurement sessions into single report-ready figures:

  * week 5 v2  -> 0.02 / 0.04 / 0.06 %  (sparse regime, focused anchor)
  * week 4 v2  -> 0.15 / 0.30 / 0.45 / 0.56 %  (dense/compact plateau)

Everything is rebuilt from the saved per-run `radius_*.csv` and the two
`fractalD_summary.csv` files -- NO video decode, so this runs in seconds and
is fully reproducible from what is committed to git.

Deliverables (agreed with the report split, Nir = Results):
  1. fill_fraction_vs_conc.png  -- sanity check: deposit occupancy of the
     enclosing disc vs concentration (expected monotone-increasing).
  2. D_vs_concentration.png     -- fractal dimension vs concentration, BOTH
     weeks, BOX-COUNTING ONLY (mass-radius estimator dropped by agreement).
  3. growth_rate_vs_conc.png    -- mean late-time front speed vs concentration.
  4. R_and_dRdt_grid.png        -- per-concentration R(t) with its dR/dt, one
     clean panel per run + a combined overlay (polished replacement for the
     7-panel draft).

Run:  python scripts/report_figures.py
"""

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # experiments/dla-concentration-final
EXP = ROOT.parent                       # experiments/
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# publication defaults: crisp type, thin spines, no figure titles (each figure
# is captioned in the report as a figure+caption combo).
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 18,
    "axes.labelpad": 9,
    "legend.fontsize": 13.5,
    "legend.framealpha": 0.92,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "axes.linewidth": 1.1,
    "axes.xmargin": 0.06,
    "axes.ymargin": 0.10,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "lines.linewidth": 2.2,
    "lines.markersize": 9,
    "figure.constrained_layout.use": False,
})

# annotation (data-label) size, scaled with the rest
LABEL_FS = 13


def save(fig, stem):
    """Write both a 300-dpi PNG (for embeds/review) and a vector PDF
    (for the manuscript). No titles are set anywhere -- captions live in
    the report."""
    png = FIGS / f"{stem}.png"
    fig.savefig(png)
    fig.savefig(FIGS / f"{stem}.pdf")
    plt.close(fig)
    return png

W4 = EXP / "week4-dla-concentration" / "version 2"
W5 = EXP / "week5-dla-concentration" / "version 2"

# raw videos live OUTSIDE git; overridable via env. Only needed to (re)grab the
# D-vs-conc snapshot crops -- every other figure is CSV-only.
VDIR4 = Path(os.environ.get("WEEK4_VIDEO_DIR",
             r"C:\dev\brownian-motion\experiments\week4-dla-no-shlomo"))
VDIR5 = Path(os.environ.get("WEEK5_VIDEO_DIR",
             r"C:\dev\brownian-motion\experiments\week5-dla-concentration\raw-videos"))
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
CROPS = FIGS / "crops"

# one row per run, ascending concentration. `csv` = committed per-run radius
# CSV; `video` = raw clip; `tmeas` = frame time the box-counting D was grounded
# on (= t_measured_s in each fractalD_summary.csv, the fully-developed frame).
RUNS = [
    dict(conc=0.02, week=5, csv=W5 / "data" / "radius_run1_c0.02.csv",
         video=VDIR5 / "run1_0.02con.mov", tmeas=403),
    dict(conc=0.04, week=5, csv=W5 / "data" / "radius_run2_c0.04.csv",
         video=VDIR5 / "run 2 conc 0.04.mov", tmeas=596),
    dict(conc=0.06, week=5, csv=W5 / "data" / "radius_run3_c0.06.csv",
         video=VDIR5 / "run3_0.06C.mov", tmeas=413),
    dict(conc=0.15, week=4, csv=W4 / "data" / "radius_run4_c0.15.csv",
         video=VDIR4 / "run4_0.15.mov", tmeas=212),
    dict(conc=0.30, week=4, csv=W4 / "data" / "radius_run3_c0.30.csv",
         video=VDIR4 / "run 3 0.3.mov", tmeas=138),
    dict(conc=0.45, week=4, csv=W4 / "data" / "radius_run2_c0.45.csv",
         video=VDIR4 / "run 2 0.45 concen.mov", tmeas=148),
    dict(conc=0.56, week=4, csv=W4 / "data" / "radius_run1_c0.56.csv",
         video=VDIR4 / "run 1 0.56 Concertation.mov", tmeas=198),
]

# systematic floor on box-counting D (the per-frame std is not recoverable
# from the summary CSV -- it needs the video frames -- so we quote the
# pipeline's documented systematic floor; see README).
D_SYS_FLOOR = 0.03


# --------------------------------------------------------------- loaders ---

def load_run(run):
    """Read one radius_*.csv -> dict of arrays + px_per_mm."""
    ppm = None
    body = []
    for line in open(run["csv"]):
        if line.startswith("#"):
            if "px_per_mm" in line:
                ppm = float(line.split("=")[1].split("+/-")[0])
            continue
        body.append(line)
    r = list(csv.DictReader(body))
    g = lambda k: np.array([float(x[k]) for x in r])
    out = dict(run)
    out.update(t=g("t_s"), M=g("M_px"), Rc=g("circ_R_px"), edge=g("edge"),
               ppm=ppm)
    return out


def load_boxcount_D():
    """conc -> box-counting D, from both weeks' fractalD_summary.csv."""
    d = {}
    for wk in (W4, W5):
        with open(wk / "data" / "fractalD_summary.csv") as fh:
            for row in csv.DictReader(fh):
                d[round(float(row["conc"]), 2)] = float(row["D_boxcount"])
    return d


# focused runs = the reliable bucket (structure optically resolved); the
# defocused runs are not reliably measurable and are shown greyed / pending.
FOCUSED_CONCS = (0.02, 0.04, 0.06, 0.15)
DEFOCUSED_CONCS = (0.30, 0.45, 0.56)


def load_reliable():
    """conc -> (D, uncertainty) for the focused runs, from the verified
    reliable-bucket CSV (box-counting on the faithful mask, window-stability
    checked)."""
    d = {}
    p = ROOT / "data" / "fractalD_reliable.csv"
    for row in csv.DictReader(l for l in open(p) if not l.startswith("#")):
        d[round(float(row["conc"]), 2)] = (float(row["D_reliable"]),
                                           float(row["unc"]))
    return d


# ------------------------------------------------------- small utilities ---

def smooth(y, k=7):
    """centred moving average with edge-replicated padding, so the boundary
    frames are not pulled toward zero (a plain 'same' convolution divides the
    ends by the full window and fabricates a drop at the last frame)."""
    y = np.asarray(y, float)
    if len(y) < 2:
        return y
    k = min(k, len(y) | 1)
    h = k // 2
    yp = np.pad(y, h, mode="edge")
    ker = np.ones(k) / k
    return np.convolve(yp, ker, mode="valid")


def edge_free(run):
    """boolean mask: deposit present and not yet touching the frame border."""
    return (run["edge"] == 0) & (run["M"] > 0)


def growth_rate_umps(run):
    """Late-time linear front speed [um/s] from a straight-line fit of the
    enclosing radius over the edge-free frames past the early transient.
    Returns (rate, stderr) in um/s."""
    m = edge_free(run)
    t, R = run["t"][m], run["Rc"][m] / run["ppm"]   # s, mm
    if len(t) < 6:
        return np.nan, np.nan
    Rmax = R.max()
    # skip the curved nucleation transient: keep the upper ~0.35..0.98 of size
    sel = (R >= 0.35 * Rmax) & (R <= 0.98 * Rmax)
    if sel.sum() < 5:
        sel = np.ones(len(t), bool)
    tt, RR = t[sel], R[sel]
    A = np.vstack([tt, np.ones_like(tt)]).T
    coef, res, *_ = np.linalg.lstsq(A, RR, rcond=None)
    dof = max(len(tt) - 2, 1)
    rv = (res[0] / dof) if len(res) else 0.0
    cov = rv * np.linalg.inv(A.T @ A)
    return coef[0] * 1000.0, float(np.sqrt(cov[0, 0]) * 1000.0)   # mm/s -> um/s


def fill_fraction(run):
    """Deposit occupancy of the enclosing disc, phi = M / (pi R^2), evaluated
    over the edge-free frames. Returns (median, lo16, hi84) [dimensionless].
    phi is size-dependent (fractal M~R^D), so this is the occupancy at each
    run's own accessible cluster size -- see README caveat."""
    m = edge_free(run)
    R, M = run["Rc"][m], run["M"][m]
    ok = R > 0
    phi = M[ok] / (np.pi * R[ok] ** 2)
    # focus on the developed cluster: the largest half of edge-free frames
    if len(phi) >= 8:
        R2 = R[ok]
        phi = phi[R2 >= np.median(R2)]
    return (float(np.median(phi)),
            float(np.percentile(phi, 16)),
            float(np.percentile(phi, 84)))


# --------------------------------------------------------------- colours ---

CONCS = [r["conc"] for r in RUNS]
NORM = Normalize(vmin=min(CONCS), vmax=max(CONCS))
CMAP = cm.viridis
color = lambda c: CMAP(NORM(c))


def dark(rgba, f=0.72):
    """Darken a colour toward black (for borders/labels on white — the bright
    viridis yellow at 0.56 % is illegible otherwise)."""
    r, g, b, a = rgba
    return (r * f, g * f, b * f, a)


# --------------------------------------------------------------- figures ---

def fig_fill_fraction(runs):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    concs, med, lo, hi = [], [], [], []
    for r in runs:
        m, l, h = fill_fraction(r)
        concs.append(r["conc"]); med.append(m); lo.append(l); hi.append(h)
    concs = np.array(concs); med = np.array(med)
    yerr = np.vstack([med - np.array(lo), np.array(hi) - med])
    ax.errorbar(concs, med, yerr=yerr, fmt="o-", ms=6, capsize=4, lw=1.4,
                color="C0", label="median over edge-free frames\n(band = 16-84 pct)")
    for i, (c, m) in enumerate(zip(concs, med)):
        dy = 14 if i % 2 == 0 else -22     # stagger to avoid label collisions
        ax.annotate(f"{m:.2f}", (c, m), textcoords="offset points",
                    xytext=(9, dy), fontsize=LABEL_FS, color="C0")
    ax.set_xlabel("CuSO$_4$ concentration [%]")
    ax.set_ylabel(r"occupancy  $\phi = M / \pi R^2$")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    out = save(fig, "fill_fraction_vs_conc")
    return out, list(zip(concs, med, lo, hi))


def _plot_D(ax, reliable):
    """Draw the reliable-bucket D-vs-concentration scatter (focused runs only,
    effective box-counting dimension), with the defocused range greyed out as
    pending. Shared by the standalone figure and the with-crops composite."""
    concs = sorted(reliable)
    D = [reliable[c][0] for c in concs]
    U = [reliable[c][1] for c in concs]
    ax.errorbar(concs, D, yerr=U, fmt="o", ms=9, capsize=4, lw=1.6,
                color="C0", label="reliable D (focused runs)")
    ax.plot(concs, D, "-", color="C0", alpha=0.4, lw=1.4)
    ax.axhline(1.71, color="k", ls="--", lw=1.4, label="2D DLA theory: 1.71")
    for i, (c, d) in enumerate(zip(concs, D)):
        dy = 14 if i % 2 == 0 else -20      # stagger to avoid label collisions
        ax.annotate(f"{d:.2f}", (c, d), textcoords="offset points",
                    xytext=(9, dy), fontsize=LABEL_FS, color="C0")
    # defocused runs: not reliably measurable -> greyed placeholder region
    ax.axvspan(0.22, 0.62, color="0.6", alpha=0.13)
    ax.text(0.42, 1.66, "defocused runs\n(0.30 / 0.45 / 0.56 %)\nD not yet reliable",
            ha="center", va="center", color="0.4", fontsize=12, style="italic")
    ax.set_xlabel("CuSO$_4$ concentration [%]")
    ax.set_ylabel("effective fractal dimension  D")
    ax.set_xlim(-0.01, 0.62); ax.set_ylim(1.45, 2.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    return concs, D


def fig_D_vs_conc(reliable):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    concs, D = _plot_D(ax, reliable)
    out = save(fig, "D_vs_concentration")
    return out, list(zip(concs, D))


# ------------------------------------------------ snapshot crops (needs video)

def _geom_at(run):
    """(cx, cy, R_px, px_per_mm) of the enclosing circle at the run's grounded
    frame time, read straight from the committed radius CSV."""
    ppm, rows = None, []
    for line in open(run["csv"]):
        if line.startswith("#"):
            if "px_per_mm" in line:
                ppm = float(line.split("=")[1].split("+/-")[0])
            continue
        rows.append(line)
    r = list(csv.DictReader(rows))
    t = np.array([float(x["t_s"]) for x in r])
    j = int(np.argmin(np.abs(t - run["tmeas"])))
    g = lambda k: float(r[j][k])
    return g("circ_cx_px"), g("circ_cy_px"), g("circ_R_px"), ppm


def grab_crop(run, pad=1.15):
    """ffmpeg-extract the grounded frame and crop a square (side = 2*pad*R)
    centred on the deposit. Returns dict(png, side_mm, ppm) or None if the
    video is unavailable. Crop geometry comes from committed CSV data."""
    if not run["video"].exists():
        return None
    CROPS.mkdir(parents=True, exist_ok=True)
    cx, cy, R, ppm = _geom_at(run)
    tmp = CROPS / f"_full_c{run['conc']:.2f}.png"
    cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-y",
           "-ss", str(run["tmeas"]), "-i", str(run["video"]),
           "-frames:v", "1", str(tmp)]
    subprocess.run(cmd, check=True)
    img = plt.imread(tmp)                       # HxWx(3/4), float 0..1
    H, W = img.shape[:2]
    half = int(round(pad * R))
    x0, x1 = max(0, int(cx) - half), min(W, int(cx) + half)
    y0, y1 = max(0, int(cy) - half), min(H, int(cy) + half)
    crop = img[y0:y1, x0:x1, :3]
    png = CROPS / f"crop_c{run['conc']:.2f}.png"
    plt.imsave(png, crop)
    tmp.unlink(missing_ok=True)
    return dict(png=png, side_mm=(x1 - x0) / ppm, ppm=ppm)


def fig_D_with_crops(runs, reliable):
    """Composite: the reliable D-vs-conc plot on top, then the 7 grounded-frame
    crops in a 4-over-3 grid. Focused crops are captioned with their reliable D;
    defocused crops are marked 'defocused'."""
    # ensure crops exist (grab from video if present, else reuse saved PNGs)
    crops = {}
    for r in runs:
        got = grab_crop(r)
        if got is None:
            png = CROPS / f"crop_c{r['conc']:.2f}.png"
            got = dict(png=png, side_mm=None, ppm=None) if png.exists() else None
        crops[r["conc"]] = got
    missing = [r["conc"] for r in runs if crops[r["conc"]] is None]
    if missing:
        print(f"[crops] no video/PNG for conc {missing} -- skipping composite")
        return None

    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.35, 1, 1],
                  hspace=0.32, wspace=0.35)
    ax_plot = fig.add_subplot(gs[0, :])
    _plot_D(ax_plot, reliable)

    order = sorted(runs, key=lambda r: r["conc"])
    # row 1: first 4 (each spans 3 of 12 cols); row 2: last 3 (each spans 4)
    slots = ([ (1, slice(3 * i, 3 * i + 3)) for i in range(4) ] +
             [ (2, slice(4 * i, 4 * i + 4)) for i in range(3) ])
    for run, (row, cols) in zip(order, slots):
        c = run["conc"]
        ax = fig.add_subplot(gs[row, cols])
        info = crops[c]
        img = plt.imread(info["png"])
        smm = info["side_mm"]
        if smm:                                  # real-units axes + 1 mm bar
            ax.imshow(img, extent=[0, smm, 0, smm])
            x0 = smm * 0.08
            ax.plot([x0, x0 + 1.0], [smm * 0.08, smm * 0.08], "-", color="w",
                    lw=3, solid_capstyle="butt")
            ax.plot([x0, x0 + 1.0], [smm * 0.08, smm * 0.08], "-", color="k",
                    lw=1.2, solid_capstyle="butt")
            ax.text(x0 + 0.5, smm * 0.11, "1 mm", color="w", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
        else:
            ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        edge = dark(color(c))
        focused = c in reliable
        for s in ax.spines.values():
            s.set_edgecolor(edge if focused else "0.6")
            s.set_linewidth(3)
        if focused:
            title = f"{c:.2f} %   D = {reliable[c][0]:.2f}"
            tcol = edge
        else:
            title = f"{c:.2f} %   defocused"
            tcol = "0.5"
        ax.set_title(title, fontsize=14, color=tcol, fontweight="bold", pad=5)
        if not focused:                      # grey wash over the unreliable crops
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       color="white", alpha=0.4, zorder=5))
    return save(fig, "D_vs_concentration_with_crops")


def fig_growth_rate(runs):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    concs, rate, err = [], [], []
    for r in runs:
        g, e = growth_rate_umps(r)
        concs.append(r["conc"]); rate.append(g); err.append(e)
    concs = np.array(concs)
    ax.errorbar(concs, rate, yerr=err, fmt="o-", ms=6, capsize=4, lw=1.4,
                color="C3", label="late-time linear front (fit $\\pm$1$\\sigma$)")
    for i, (c, g) in enumerate(zip(concs, rate)):
        dy = 14 if i % 2 == 0 else -22     # stagger to avoid label collisions
        ax.annotate(f"{g:.1f}", (c, g), textcoords="offset points",
                    xytext=(9, dy), fontsize=LABEL_FS, color="C3")
    ax.set_xlabel("CuSO$_4$ concentration [%]")
    ax.set_ylabel("mean growth rate  dR/dt [µm/s]")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    out = save(fig, "growth_rate_vs_conc")
    return out, list(zip(concs, rate, err))


def fig_R_dRdt_grid(runs):
    """One clean panel per concentration: R(t) [mm] on the left axis,
    dR/dt [um/s] on the right, plus a combined R(t) overlay in the 8th slot."""
    fig, axes = plt.subplots(4, 2, figsize=(17, 20))
    axes = axes.ravel()
    for ax, r in zip(axes, runs):
        m = edge_free(r)
        t = r["t"][m]
        R = r["Rc"][m] / r["ppm"]                    # mm
        Rs = smooth(R, 9)
        # derivative of the smoothed radius, in um/s
        dR = np.gradient(Rs, t) * 1000.0
        c = color(r["conc"])
        ax.plot(t, R, color="0.75", lw=0.8, label="R (raw)")
        ax.plot(t, Rs, color=c, lw=2.0, label="R (smoothed)")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("enclosing R [mm]", color=c)
        ax.tick_params(axis="y", labelcolor=c)
        # panel identifier (data label, not a title) in the top-right corner
        ax.text(0.97, 0.06, f"{r['conc']:.2f} %", transform=ax.transAxes,
                ha="right", va="bottom", fontweight="bold", fontsize=17, color=c)
        ax.grid(alpha=0.25)
        axd = ax.twinx()
        axd.plot(t, dR, color="C1", lw=1.1, alpha=0.85, label="dR/dt")
        axd.set_ylabel("dR/dt [µm/s]", color="C1")
        axd.tick_params(axis="y", labelcolor="C1")
        axd.set_ylim(bottom=0)
        # single combined legend per panel
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axd.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=12, loc="upper left", framealpha=0.9)

    # 8th slot: all runs' R(t) overlaid, coloured by concentration
    ax = axes[7]
    for r in runs:
        m = edge_free(r)
        t = r["t"][m]; R = smooth(r["Rc"][m] / r["ppm"], 9)
        ax.plot(t, R, color=color(r["conc"]), lw=1.8, label=f"{r['conc']:.2f} %")
    ax.set_xlabel("time [s]"); ax.set_ylabel("enclosing R [mm]")
    ax.text(0.97, 0.06, "all runs", transform=ax.transAxes, ha="right",
            va="bottom", fontweight="bold", fontsize=17, color=color(0.30))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12.5, title="CuSO$_4$", title_fontsize=13, ncol=2, loc="upper left")

    fig.tight_layout()
    return save(fig, "R_and_dRdt_grid")


# ------------------------------------------------------------------ main ---

def main():
    runs = [load_run(r) for r in RUNS]
    reliable = load_reliable()

    f1, fill = fig_fill_fraction(runs)
    f2, Dvals = fig_D_vs_conc(reliable)
    f3, rates = fig_growth_rate(runs)
    f4 = fig_R_dRdt_grid(runs)
    f5 = fig_D_with_crops(runs, reliable)

    print("\n=== fill fraction phi = M/(pi R^2) (median, 16pct, 84pct) ===")
    for c, m, lo, hi in fill:
        print(f"  {c:.2f}% : {m:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\n=== reliable fractal D vs conc (focused runs) ===")
    for c, d in Dvals:
        print(f"  {c:.2f}% : D = {d:.3f}")

    print("\n=== growth rate [um/s] ===")
    for c, g, e in rates:
        print(f"  {c:.2f}% : {g:.1f} +/- {e:.1f}")

    print("\nfigures ->")
    for f in (f1, f2, f3, f4, f5):
        if f:
            print(f"  {f}")


if __name__ == "__main__":
    main()
