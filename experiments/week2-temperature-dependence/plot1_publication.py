"""
plot1_publication.py -- clean, publication-ready MSD-vs-lag grid (Figure 1).

The COMPLEMENT to kb_grid_publication.py (Figure 2): it shows where the D values
come from. ONE representative run per temperature (the most-populated gate-passing
run at each T) -- the grid reads as a temperature sweep, one cell per temperature.
Each panel shows that run's ensemble MSD <r^2>(tau) as points with per-lag error
bars (bead-to-bead SE), the ensemble fit <r^2> = 4 D tau + c (solid in the
short-lag window, dashed extrapolation) with a +/-1 sigma band, and the ensemble
<D> annotated.

Minimal ink: the ensemble MSD points (+/- SE), the fit line + band, one text line
(run, T, <D>, n) -- NO faint per-particle "ghost" curves. Reuses
kb_grid.analyse_run for the gate-passing set and pipeline.msd for the MSD, so the
particle set is identical to Fig 2.

Writes figures/plot1_publication.png (+ .pdf).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline import paths, physics
from pipeline import msd as M
import kb_grid

FIT_LAG = 30        # short-lag fit window (frames) -- matches msd.py
PLOT_LAG = 56       # show a bit past the fit window

# Representative run per temperature defaults to the most-populated gate-passing
# run; stems listed here are PINNED as their temperature's representative instead.
# run16 (30.3 C) has the most beads but carries the table-collision disturbance,
# so the cleaner run15 is pinned for 30.3 C.
PINNED_REPS = {"run15"}


def ensemble_and_curves(stem, pids, mpp, dt):
    """Per-particle MSD curves (um^2 vs s) + the n_pairs-weighted ensemble and
    its per-lag uncertainty.

    The ensemble pools beads of DIFFERENT radii (hence different D), so the
    honest per-lag error on the ensemble mean is the bead-to-bead spread,
    sigma_ens(tau) = std_beads(MSD_i(tau)) / sqrt(n_beads) -- the same scatter
    that sets the annotated <D> +/- SE (the fit-covariance SE collapses to ~0
    because the overlapping-pair ensemble points are strongly correlated). It
    grows with tau as the different-D beads fan out, which is exactly why the
    fit is restricted to the short-lag window.
    """
    traj = pd.read_csv(os.path.join(paths.out_dir(stem, make=False), "trajectory.csv"))
    curves = []
    ens_sum, ens_w, per_lag = {}, {}, {}
    for pid in pids:
        g = traj[traj["particle"] == pid].sort_values("frame")
        lag, msd_px2, npair, _dx, _dy = M.per_bead_msd(
            g["frame"].values, g["x"].values, g["y"].values, PLOT_LAG)
        if len(lag) < 3:
            continue
        m_um2 = msd_px2 * mpp * mpp
        curves.append((lag * dt, m_um2))
        for L, m, w in zip(lag, m_um2, npair):
            ens_sum[L] = ens_sum.get(L, 0.0) + m * w
            ens_w[L] = ens_w.get(L, 0.0) + w
            per_lag.setdefault(L, []).append(m)
    Ls = np.array(sorted(ens_sum))
    ens = np.array([ens_sum[L] / ens_w[L] for L in Ls])
    # per-lag bead-to-bead SE of the ensemble mean
    ens_se = np.array([
        (np.std(per_lag[L], ddof=1) / np.sqrt(len(per_lag[L])))
        if len(per_lag[L]) > 1 else np.nan for L in Ls])
    et = Ls * dt
    fm = Ls <= FIT_LAG
    if fm.sum() >= 2:
        (slope, intercept), cov = np.polyfit(et[fm], ens[fm], 1, cov=True)
        se_D = float(np.sqrt(max(cov[0, 0], 0))) / 4.0
    else:
        slope = intercept = se_D = np.nan
    return curves, et, ens, ens_se, slope / 4.0, intercept, se_D, FIT_LAG * dt


def main():
    mpp = paths.load_scale() or 0.14381
    runs = sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:]))
    res = []
    for stem in runs:
        if stem in kb_grid.EXCLUDED_RUNS:
            continue
        r = kb_grid.analyse_run(stem, mpp)
        if r is not None and r["n"] >= 3:
            res.append(r)
    # ONE representative run per temperature -- the most-populated gate-passing
    # run at each T (same rule finalize.py uses for the per-bead diagnostics),
    # unless a run is PINNED for that temperature, so the grid reads as a clean
    # temperature sweep, one cell per temperature.
    from collections import defaultdict
    byT = defaultdict(list)
    for r in res:
        byT[r["T"]].append(r)
    res = []
    for _, grp in sorted(byT.items()):
        pinned = [r for r in grp if r["run"] in PINNED_REPS]
        res.append(pinned[0] if pinned else max(grp, key=lambda r: r["n"]))

    plt.rcParams.update({
        "font.size": 11, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    })
    BLUE, RED = "#2b6cb0", "#c1272d"
    # one row: one cell per temperature, side by side (the temperature sweep)
    ncols = len(res)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 4.0),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(res):]:
        ax.set_visible(False)

    panels, devs = [], []
    for stem_res in res:
        stem = stem_res["run"]
        fps = paths.fps_of(paths.video_for_run(stem)) or 9.30
        pids = stem_res["fit"]["particle"].astype(int).tolist()
        cu, et, ens, ens_se, D_ens, c, _se_cov, tfit = ensemble_and_curves(stem, pids, mpp, 1.0 / fps)
        Di = stem_res["fit"]["D_um2_s"].values
        D_med = float(np.median(Di))                         # Fig-2 estimator
        # honest SE: bead-to-bead scatter / sqrt(n) (the polyfit-cov SE collapses
        # to ~0 because the ensemble-MSD points are strongly correlated)
        se_D = float(np.std(Di, ddof=1) / np.sqrt(len(Di))) if len(Di) > 1 else np.nan
        devs.append(abs(D_ens - D_med) / D_med)
        panels.append((stem_res, cu, et, ens, ens_se, D_ens, c, se_D, tfit))
    tmax = PLOT_LAG / (paths.fps_of(paths.video_for_run(res[0]["run"])) or 9.30)
    ymax = max(np.nanmax((ens + np.nan_to_num(ens_se))[et <= tmax])
               for _, _, et, ens, ens_se, *_ in panels) * 1.06

    for ax, (r, cu, et, ens, ens_se, D_ens, c, se_D, tfit) in zip(axes.flat, panels):
        es = et <= tmax
        # ensemble MSD with per-lag bead-to-bead SE error bars
        ax.errorbar(et[es], ens[es], yerr=ens_se[es], fmt="o", ms=3.6, color=BLUE,
                    mec="white", mew=0.4, ecolor=BLUE, elinewidth=0.7, capsize=1.3,
                    zorder=4)
        yfit = 4 * D_ens * tfit + c
        # +/-1 sigma band of the fit slope (<D> +/- SE) over the fit window
        tb = np.linspace(0, tfit, 40)
        lb = 4 * D_ens * tb + c
        db = 4 * se_D * tb
        ax.fill_between(tb, lb - db, lb + db, color=RED, alpha=0.13, lw=0, zorder=3)
        ax.plot([0, tfit], [c, yfit], "-", color=RED, lw=2.2, zorder=5)        # fit window: solid
        ax.plot([tfit, tmax], [yfit, 4 * D_ens * tmax + c], "--", color=RED,    # beyond: dashed
                lw=1.4, dashes=(4, 3), zorder=5)
        ax.axvline(tfit, color="0.6", lw=0.8, ls=":", zorder=1)                 # fit-window edge
        ax.set_xlim(0, tmax); ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=9, length=3)
        ax.grid(True, color="0.92", lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.text(0.05, 0.95, f"{r['run']}  ·  {r['T']:.1f} °C",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color="0.35")
        ax.text(0.05, 0.85, rf"$\langle D\rangle = {D_ens:.3f}\pm{se_D:.3f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=10.5, color=RED)
        ax.text(0.05, 0.75, rf"$n = {r['n']}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.5, color="0.45")

    for i in range(nrows):
        axes[i][0].set_ylabel(r"$\langle r^2\rangle$   [$\mu$m$^2$]", fontsize=11)
    for j in range(ncols):
        axes[nrows - 1][j].set_xlabel(r"lag time  $\tau$   [s]", fontsize=11)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg = [Line2D([0], [0], marker="o", color=BLUE, mfc=BLUE, mec="white", ms=7,
                  ls="none", label=r"ensemble MSD $\pm$ SE"),
           Line2D([0], [0], color=RED, lw=2.2, label=r"fit $\langle r^2\rangle=4D\tau+c$"),
           Patch(facecolor=RED, alpha=0.2, label=r"$\pm1\sigma$ ($\langle D\rangle$ SE)"),
           Line2D([0], [0], color=RED, lw=1.4, ls="--", label="extrapolation (excl. from fit)")]
    fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False,
               fontsize=9.0, bbox_to_anchor=(0.5, -0.04))
    # explanatory footnote intentionally omitted -- it lives in the report caption
    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    out = os.path.join(paths.FIGURES_DIR, "plot1_publication.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    try:
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight"); pdf = " (+ .pdf)"
    except PermissionError:
        pdf = " (.pdf locked)"
    plt.close(fig)
    print(f"wrote {out}{pdf}; {len(res)} gate-passing runs (units D in um^2/s)")


if __name__ == "__main__":
    main()
