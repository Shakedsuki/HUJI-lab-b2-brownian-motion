#!/usr/bin/env python3
"""Combined report figures for the DLA-vs-CuSO4-concentration experiment.

Merges the two measurement sessions into single report-ready figures:

  * week 5 v2  -> 0.02 / 0.04 / 0.06 %  (sparse regime, focused anchor)
  * week 4 v2  -> 0.15 / 0.45 / 0.56 %  (dense/compact plateau)

0.30 % is EXCLUDED from the dataset: its video is defocused (blur sigma
2.5-3 px) and the deposit offers <0.6 decades of scaling range at every
moment of the run -- below the box-counting estimator's validated regime;
all salvage routes failed their controls (NOTES_defocused_runs_recovery.md).
0.45 / 0.56 % were recovered by the same audit and carry reliable D values
in data/fractalD_reliable.csv.

Everything is rebuilt from the saved per-run `radius_*.csv` and the two
`fractalD_summary.csv` files -- NO video decode, so this runs in seconds and
is fully reproducible from what is committed to git.

Deliverables (agreed with the report split, Nir = Results):
  1. fill_fraction_vs_conc.png  -- sanity check: deposit occupancy of the
     enclosing disc vs concentration (expected monotone-increasing).
  2. D_vs_concentration.png     -- fractal dimension vs concentration, BOTH
     weeks, BOX-COUNTING ONLY (mass-radius estimator dropped by agreement).
  3. growth_rate_vs_conc.png    -- mean late-time front speed vs concentration.
  4. R_vs_t_grid.png            -- per-concentration R(t), one clean panel
     per run over the FULL run (plateau visible; dR/dt twin axis and the
     all-runs overlay dropped -- too crowded).

Run:  python scripts/report_figures.py
"""

import csv
import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
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
    # 0.30 % (run 3 0.3.mov) intentionally absent: excluded for defocus --
    # see module docstring.
    # lamp=True: heat lamp on for the WHOLE run (paper R/B ~ 1.40 vs 0.95
    # neutral) -> thermal convection may inflate its transport-limited
    # numbers; flagged with an asterisk on the growth-rate figure. The only
    # affected run: 0.56's lamp only comes on at t ~ 400 s, after every
    # measured window; all other runs are neutral throughout.
    dict(conc=0.45, week=4, csv=W4 / "data" / "radius_run2_c0.45.csv",
         video=VDIR4 / "run 2 0.45 concen.mov", tmeas=148, lamp=True),
    dict(conc=0.56, week=4, csv=W4 / "data" / "radius_run1_c0.56.csv",
         video=VDIR4 / "run 1 0.56 Concertation.mov", tmeas=198),
]

EXCLUDED_CONC = 0.30    # annotated on the D figure so the gap is explained
LAMP_CONCS = tuple(r["conc"] for r in RUNS if r.get("lamp"))
LAMP_LABEL = "heat-lamp on during run"


def lamp_star(ax, x, y, dx=-13, dy=2):
    """The asterisk marking a heat-lamp-affected point, for prose reference."""
    ax.annotate("*", (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=26, color="k", ha="center", va="center",
                fontweight="bold")


def conc_axis(ax, concs):
    """Linear concentration axis (collaborator request: with only six fairly
    spread points a log axis is unwarranted). Default numeric ticks -- a
    labelled tick at every measured value would collide at 0.02/0.04/0.06;
    the per-point data labels carry the exact values instead."""
    ax.set_xlim(-0.02, max(concs) * 1.08)
    ax.set_xlabel("CuSO$_4$ concentration [%]")


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
    out.update(t=g("t_s"), M=g("M_px"), Rc=g("circ_R_px"),
               Rraw=g("circ_R_raw_px"), edge=g("edge"), ppm=ppm)
    return out


def load_reliable():
    """conc -> (D, uncertainty) for every run in the dataset, from the
    verified reliable-bucket CSV (box-counting on the faithful mask,
    window-stability checked; 0.45/0.56 recovered per the defocus audit)."""
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


def detection_settled(run, window_s=60.0, tol_px=2.0):
    """boolean mask: past the initial detection transient.

    Early on, segmentation admits the young deposit in a batch: the UNCAPPED
    envelope steps super-physically (mm per sample) and the capped/reported R
    replays the step as a fake 90 um/s ramp. Both series are corrupted until
    they agree again (the hand-off). Frames before the last early frame where
    the cap still lags the raw envelope are display/fit-worthless."""
    lag = (run["Rraw"] - run["Rc"] > tol_px) & (run["t"] < window_s)
    idx = np.where(lag)[0]
    return run["t"] > (run["t"][idx[-1]] if len(idx) else -np.inf)


def growth_rate_umps(run):
    """Late-time linear front speed [um/s] from a straight-line fit of the
    UNCAPPED enclosing radius over the edge-free frames past the detection
    transient (the capped series' fake catch-up ramp would otherwise leak
    ~+5% into the 0.45% fit). Returns (rate, stderr) in um/s."""
    m = edge_free(run) & detection_settled(run)
    t, R = run["t"][m], run["Rraw"][m] / run["ppm"]   # s, mm
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

CONCS = sorted(r["conc"] for r in RUNS)
CMAP = cm.viridis
# colour by RANK, not by value: the concentrations are log-spaced, so a
# value-linear mapping squeezes 0.02/0.04/0.06 into near-identical purples.
_POS = {c: i / (len(CONCS) - 1) for i, c in enumerate(CONCS)}
color = lambda c: CMAP(_POS[round(c, 2)])


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
                color="C0", label="Median")
    for i, (c, m) in enumerate(zip(concs, med)):
        dy = 14 if i % 2 == 0 else -22     # stagger to avoid label collisions
        ax.annotate(f"{m:.2f}", (c, m), textcoords="offset points",
                    xytext=(9, dy), fontsize=LABEL_FS, color="C0")
    for c, m in zip(concs, med):
        if c in LAMP_CONCS:
            lamp_star(ax, c, m)
    ax.plot([], [], ls="none", marker="$*$", ms=13, color="k",
            label=LAMP_LABEL)
    conc_axis(ax, concs)
    ax.set_ylabel(r"occupancy  $\phi = M / \pi R^2$")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    out = save(fig, "fill_fraction_vs_conc")
    return out, list(zip(concs, med, lo, hi))


# 0.30 % is excluded from the reliable-D fit (defocused, <0.6 decades of
# scaling range; see module docstring). It still has 5 box-counting estimates
# on record -- same estimator as every other point, just individually
# untrustworthy -- from single-frame, deblurred, and time-ensemble attempts
# (fractalD_defocused.csv, fractalD_deblur.csv, salvage_c030.csv route A).
# The mass-radius salvage route (route C) is NOT included: that estimator was
# dropped project-wide, so mixing it in here would compare apples to oranges.
# Their mean +/- sample std is plotted as an explicitly-flagged X, not a
# measurement -- it locates the excluded slot without pretending it's data.
EXCLUDED_D_ESTIMATES = [1.9831, 1.9869, 1.9638, 1.9009, 2.0001]
EXCLUDED_D_MEAN = float(np.mean(EXCLUDED_D_ESTIMATES))
EXCLUDED_D_STD = float(np.std(EXCLUDED_D_ESTIMATES, ddof=1))


def _plot_D(ax, reliable):
    """Draw the reliable D-vs-concentration scatter (effective box-counting
    dimension, window-stability verified), plus the excluded 0.30 % slot
    marked with an X (mean of its unreliable box-counting estimates, not a
    measurement). Shared by the standalone figure and the with-crops
    composite."""
    concs = sorted(reliable)
    D = [reliable[c][0] for c in concs]
    U = [reliable[c][1] for c in concs]
    ax.errorbar(concs, D, yerr=U, fmt="o", ms=9, capsize=4, lw=1.6,
                color="C0", label="box-counting D")
    ax.plot(concs, D, "-", color="C0", alpha=0.4, lw=1.4)
    ax.axhline(1.71, color="k", ls="--", lw=1.4, label="2D DLA theory: 1.71")
    for i, (c, d) in enumerate(zip(concs, D)):
        # stagger: odd points labelled left-below, even right-above (the 0.04
        # and 0.06 points are close enough that same-side labels collide)
        dx, dy = (9, 14) if i % 2 == 0 else (-48, -22)
        ax.annotate(f"{d:.2f}", (c, d), textcoords="offset points",
                    xytext=(dx, dy), fontsize=LABEL_FS, color="C0")
    # excluded 0.30 % run: no line to its neighbours -- it is not part of
    # the fitted trend, just located on the axis.
    ax.errorbar([EXCLUDED_CONC], [EXCLUDED_D_MEAN], yerr=[EXCLUDED_D_STD],
                fmt="x", ms=13, mew=2.5, capsize=4, lw=1.6, color="0.45")
    for c in LAMP_CONCS:
        if c in reliable:
            lamp_star(ax, c, reliable[c][0])
    ax.plot([], [], ls="none", marker="$*$", ms=13, color="k",
            label=LAMP_LABEL)
    conc_axis(ax, sorted(concs + [EXCLUDED_CONC]))
    ax.set_ylabel("effective fractal dimension  D")
    ax.set_ylim(1.45, 2.07)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
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
    """Composite: the reliable D-vs-conc plot on top, then the 6 grounded-frame
    crops in a 3-over-3 grid, each captioned with its reliable D."""
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

    fig = plt.figure(figsize=(15, 15.6))
    # two independently-positioned GridSpecs: the top plot's rotated tick
    # labels + xlabel need real room below it, decoupled from the crop
    # grid's own row spacing (a single shared hspace let the D-vs-conc
    # x-axis text collide with the crop row underneath it).
    gs_plot = GridSpec(1, 1, figure=fig, top=0.98, bottom=0.68,
                        left=0.07, right=0.99)
    gs_crops = GridSpec(2, 12, figure=fig, top=0.60, bottom=0.02,
                         hspace=0.32, wspace=0.35, left=0.01, right=0.99)
    ax_plot = fig.add_subplot(gs_plot[0, 0])
    _plot_D(ax_plot, reliable)

    order = sorted(runs, key=lambda r: r["conc"])
    # split the crops over two rows on the 12-column grid (6 runs -> 3 + 3)
    top = (len(order) + 1) // 2
    bot = len(order) - top
    slots = ([(0, slice((12 // top) * i, (12 // top) * (i + 1))) for i in range(top)] +
             [(1, slice((12 // bot) * i, (12 // bot) * (i + 1))) for i in range(bot)])
    for run, (row, cols) in zip(order, slots):
        c = run["conc"]
        ax = fig.add_subplot(gs_crops[row, cols])
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
            star = "*" if run.get("lamp") else ""
            title = f"{c:.2f} %{star}   D = {reliable[c][0]:.2f}"
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
    # heat-lamp-affected run(s): asterisk for reference from the prose
    # (convection may inflate the transport-limited rate)
    for r, g in zip(runs, rate):
        if r.get("lamp"):
            lamp_star(ax, r["conc"], g)
    ax.plot([], [], ls="none", marker="$*$", ms=13, color="k",
            label=LAMP_LABEL)
    conc_axis(ax, concs)
    ax.set_ylabel("mean growth rate  dR/dt [µm/s]")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    out = save(fig, "growth_rate_vs_conc")
    return out, list(zip(concs, rate, err))


def fig_R_grid(runs):
    """3x2 grid of per-concentration R(t) panels -- radius only (the dR/dt
    twin axis and the all-runs overlay column were dropped by agreement:
    the composite was too crowded).

    The FULL run is shown, not just the edge-free segment: the late-time
    plateau (growth stalling as the deposit spans the cell) is the point of
    the figure -- truncating at frame contact left every curve looking
    linear. Frames after the deposit first touches the frame border are
    drawn dashed: from there the enclosing radius is measured against a
    partially out-of-frame deposit, so the plateau level is a lower bound."""
    nrows = (len(runs) + 1) // 2            # per-run panels in 2 columns
    fig = plt.figure(figsize=(16, 5.0 * nrows))
    gs = GridSpec(nrows, 2, figure=fig, hspace=0.30, wspace=0.24)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(runs))]
    for k, (ax, r) in enumerate(zip(axes, runs)):
        # UNCAPPED envelope, past the detection transient: the capped series'
        # fake 90 um/s ramp + hand-off drop never enter the figure
        m = (r["M"] > 0) & detection_settled(r)
        t = r["t"][m]
        Rs = smooth(r["Rraw"][m] / r["ppm"], 9)      # mm
        free = r["edge"][m] == 0
        # last edge-free index: split solid (trusted) / dashed (frame-limited)
        j = int(np.where(free)[0][-1]) if free.any() else len(t) - 1
        c = dark(color(r["conc"]))     # darkened: raw viridis yellow (0.56 %)
                                       # is barely visible on white
        # (the unsmoothed series differs from the smoothed one only at
        # staircase corners -- invisible under it, so it is not drawn)
        ax.plot(t[:j + 1], Rs[:j + 1], color=c, lw=2.0, label="in frame")
        ax.plot(t[j:], Rs[j:], color=c, lw=2.0, ls=(0, (4, 2)), alpha=0.8,
                label="frame contact")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("enclosing R [mm]")
        # panel identifier (data label, not a title) in the corner;
        # asterisk = heat-lamp-affected run (see growth-rate figure legend)
        tag = f"{r['conc']:.2f} %" + ("*" if r.get("lamp") else "")
        ax.text(0.97, 0.06, tag, transform=ax.transAxes,
                ha="right", va="bottom", fontweight="bold", fontsize=17, color=c)
        ax.grid(alpha=0.25)
        if k == 0:      # line-style key once, first panel (colours vary only)
            ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    return save(fig, "R_vs_t_grid")


# ------------------------------------------------------------------ main ---

def main():
    runs = [load_run(r) for r in RUNS]
    reliable = load_reliable()

    f1, fill = fig_fill_fraction(runs)
    f2, Dvals = fig_D_vs_conc(reliable)
    f3, rates = fig_growth_rate(runs)
    f4 = fig_R_grid(runs)
    f5 = fig_D_with_crops(runs, reliable)

    print("\n=== fill fraction phi = M/(pi R^2) (median, 16pct, 84pct) ===")
    for c, m, lo, hi in fill:
        print(f"  {c:.2f}% : {m:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\n=== reliable fractal D vs conc (0.30 % excluded: defocus) ===")
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
