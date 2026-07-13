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
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # experiments/dla-concentration-final
EXP = ROOT.parent                       # experiments/
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

W4 = EXP / "week4-dla-concentration" / "version 2"
W5 = EXP / "week5-dla-concentration" / "version 2"

# one row per run, ascending concentration; source CSVs are per-week.
RUNS = [
    dict(conc=0.02, week=5, csv=W5 / "data" / "radius_run1_c0.02.csv"),
    dict(conc=0.04, week=5, csv=W5 / "data" / "radius_run2_c0.04.csv"),
    dict(conc=0.06, week=5, csv=W5 / "data" / "radius_run3_c0.06.csv"),
    dict(conc=0.15, week=4, csv=W4 / "data" / "radius_run4_c0.15.csv"),
    dict(conc=0.30, week=4, csv=W4 / "data" / "radius_run3_c0.30.csv"),
    dict(conc=0.45, week=4, csv=W4 / "data" / "radius_run2_c0.45.csv"),
    dict(conc=0.56, week=4, csv=W4 / "data" / "radius_run1_c0.56.csv"),
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


# --------------------------------------------------------------- figures ---

def fig_fill_fraction(runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    concs, med, lo, hi = [], [], [], []
    for r in runs:
        m, l, h = fill_fraction(r)
        concs.append(r["conc"]); med.append(m); lo.append(l); hi.append(h)
    concs = np.array(concs); med = np.array(med)
    yerr = np.vstack([med - np.array(lo), np.array(hi) - med])
    ax.errorbar(concs, med, yerr=yerr, fmt="o-", ms=6, capsize=4, lw=1.4,
                color="C0", label="median over edge-free frames\n(band = 16-84 pct)")
    ax.set_xlabel("CuSO$_4$ concentration [%]")
    ax.set_ylabel(r"occupancy  $\phi = M / \pi R^2$")
    ax.set_title("Deposit occupancy of the enclosing disc vs concentration\n"
                 "(sanity check: denser deposits at higher concentration)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out = FIGS / "fill_fraction_vs_conc.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out, list(zip(concs, med, lo, hi))


def fig_D_vs_conc(runs, Dbox):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    concs = np.array([r["conc"] for r in runs])
    D = np.array([Dbox[r["conc"]] for r in runs])
    ax.errorbar(concs, D, yerr=D_SYS_FLOOR, fmt="o", ms=7, capsize=4, lw=1.4,
                color="C0", label="box-counting D  ($\\pm$0.03 syst.)")
    ax.plot(concs, D, "-", color="C0", alpha=0.4, lw=1.2)
    ax.axhline(1.71, color="k", ls="--", lw=1.2, label="2D DLA theory: 1.71")
    ax.axhline(2.0, color="0.5", ls=":", lw=1.0, label="compact (2D): 2.0")
    for i, (c, d) in enumerate(zip(concs, D)):
        dy = 9 if i % 2 == 0 else -14      # stagger to avoid label collisions
        ax.annotate(f"{d:.2f}", (c, d), textcoords="offset points",
                    xytext=(7, dy), fontsize=8, color="C0")
    ax.set_xlabel("CuSO$_4$ concentration [%]")
    ax.set_ylabel("fractal dimension  D")
    ax.set_title("Fractal dimension vs concentration (box-counting, both sessions)")
    ax.set_ylim(1.45, 2.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = FIGS / "D_vs_concentration.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out, list(zip(concs, D))


def fig_growth_rate(runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    concs, rate, err = [], [], []
    for r in runs:
        g, e = growth_rate_umps(r)
        concs.append(r["conc"]); rate.append(g); err.append(e)
    concs = np.array(concs)
    ax.errorbar(concs, rate, yerr=err, fmt="o-", ms=6, capsize=4, lw=1.4,
                color="C3", label="late-time linear front (fit $\\pm$1$\\sigma$)")
    ax.set_xlabel("CuSO$_4$ concentration [%]")
    ax.set_ylabel("mean growth rate  dR/dt [µm/s]")
    ax.set_title("Envelope growth rate vs concentration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out = FIGS / "growth_rate_vs_conc.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out, list(zip(concs, rate, err))


def fig_R_dRdt_grid(runs):
    """One clean panel per concentration: R(t) [mm] on the left axis,
    dR/dt [um/s] on the right, plus a combined R(t) overlay in the 8th slot."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
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
        ax.set_title(f"{r['conc']:.2f} %  (week {r['week']})", fontweight="bold")
        ax.grid(alpha=0.25)
        axd = ax.twinx()
        axd.plot(t, dR, color="C1", lw=1.1, alpha=0.85, label="dR/dt")
        axd.set_ylabel("dR/dt [µm/s]", color="C1")
        axd.tick_params(axis="y", labelcolor="C1")
        axd.set_ylim(bottom=0)
        # single combined legend per panel
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axd.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)

    # 8th slot: all runs' R(t) overlaid, coloured by concentration
    ax = axes[7]
    for r in runs:
        m = edge_free(r)
        t = r["t"][m]; R = smooth(r["Rc"][m] / r["ppm"], 9)
        ax.plot(t, R, color=color(r["conc"]), lw=1.8, label=f"{r['conc']:.2f} %")
    ax.set_xlabel("time [s]"); ax.set_ylabel("enclosing R [mm]")
    ax.set_title("all concentrations — R(t) overlay", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="CuSO$_4$", ncol=2, loc="upper left")

    fig.suptitle("Enclosing-circle radius and growth rate per concentration "
                 "(edge-free frames)", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = FIGS / "R_and_dRdt_grid.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ------------------------------------------------------------------ main ---

def main():
    runs = [load_run(r) for r in RUNS]
    Dbox = load_boxcount_D()

    f1, fill = fig_fill_fraction(runs)
    f2, Dvals = fig_D_vs_conc(runs, Dbox)
    f3, rates = fig_growth_rate(runs)
    f4 = fig_R_dRdt_grid(runs)

    print("\n=== fill fraction phi = M/(pi R^2) (median, 16pct, 84pct) ===")
    for c, m, lo, hi in fill:
        print(f"  {c:.2f}% : {m:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\n=== box-counting D vs conc (both weeks) ===")
    for c, d in Dvals:
        print(f"  {c:.2f}% : D = {d:.3f} +/- {D_SYS_FLOOR:.2f}")

    print("\n=== growth rate [um/s] ===")
    for c, g, e in rates:
        print(f"  {c:.2f}% : {g:.1f} +/- {e:.1f}")

    print("\nfigures ->")
    for f in (f1, f2, f3, f4):
        print(f"  {f}")


if __name__ == "__main__":
    main()
