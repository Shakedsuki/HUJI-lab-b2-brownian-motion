"""
plot1_publication.py -- clean, publication-ready MSD-vs-lag grid (Figure 1).

The COMPLEMENT to kb_grid_publication.py (Figure 2): it shows where the D values
come from. One panel per gate-passing run (the same 10 as Fig 2); within each
panel the same gate-passing particles. Each particle's time-averaged MSD
<r^2>(tau) is drawn faint (showing the motion is diffusive -- <r^2> linear in
tau), with the ensemble fit <r^2> = 4 D tau prominent and the ensemble <D>
annotated. Each faint curve maps one-to-one onto a point in Fig 2.

Minimal ink: faint per-particle MSDs, the ensemble fit line, one text line
(run, T, <D>, n). Reuses kb_grid.analyse_run for the gate-passing set and
pipeline.msd for the MSD, so the particle set is identical to Fig 2.

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


def ensemble_and_curves(stem, pids, mpp, dt):
    """Per-particle MSD curves (um^2 vs s) + the n_pairs-weighted ensemble."""
    traj = pd.read_csv(os.path.join(paths.out_dir(stem, make=False), "trajectory.csv"))
    curves = []
    ens_sum, ens_w = {}, {}
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
    Ls = np.array(sorted(ens_sum))
    ens = np.array([ens_sum[L] / ens_w[L] for L in Ls])
    et = Ls * dt
    fm = Ls <= FIT_LAG
    slope, intercept = (np.polyfit(et[fm], ens[fm], 1) if fm.sum() >= 2
                        else (np.nan, np.nan))
    return curves, et, ens, slope / 4.0, intercept


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
    res.sort(key=lambda r: (r["T"], int(r["run"][3:])))

    plt.rcParams.update({
        "font.size": 11, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    })
    BLUE, RED = "#2b6cb0", "#c1272d"
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 6.3),
                             sharex=True, sharey=True)

    panels = []
    for stem_res in res:
        stem = stem_res["run"]
        fps = paths.fps_of(paths.video_for_run(stem)) or 9.30
        pids = stem_res["fit"]["particle"].astype(int).tolist()
        curves, et, ens, D_ens, c = ensemble_and_curves(stem, pids, mpp, 1.0 / fps)
        panels.append((stem_res, curves, et, ens, D_ens, c))
    tmax = PLOT_LAG / (paths.fps_of(paths.video_for_run(res[0]["run"])) or 9.30)
    ymax = max(np.nanmax(ens[et <= tmax]) for _, _, et, ens, _, _ in panels) * 1.08

    for ax, (r, curves, et, ens, D_ens, c) in zip(axes.flat, panels):
        for t, m in curves:                      # faint per-particle MSDs
            sel = t <= tmax
            ax.plot(t[sel], m[sel], "-", color=BLUE, lw=0.7, alpha=0.28, zorder=2)
        es = et <= tmax
        ax.plot(et[es], ens[es], "o", ms=3.6, color=BLUE, mec="white", mew=0.4,
                zorder=4)
        ax.plot([0, tmax], [c, 4 * D_ens * tmax + c], "-", color=RED, lw=2.0, zorder=5)
        ax.set_xlim(0, tmax); ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=9, length=3)
        ax.grid(True, color="0.92", lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.text(0.05, 0.95, f"{r['run']}  ·  {r['T']:.1f} °C",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color="0.35")
        ax.text(0.05, 0.85, rf"$\langle D\rangle = {D_ens:.3f}\ \mu$m$^2$s$^{{-1}}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=10.5, color=RED)
        ax.text(0.05, 0.74, rf"$n = {r['n']}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.5, color="0.45")

    for i in range(nrows):
        axes[i][0].set_ylabel(r"$\langle r^2\rangle$   [$\mu$m$^2$]", fontsize=11)
    for j in range(ncols):
        axes[nrows - 1][j].set_xlabel(r"lag time  $\tau$   [s]", fontsize=11)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=BLUE, lw=1.4, alpha=0.5, label="particle MSD"),
           Line2D([0], [0], marker="o", color="w", mfc=BLUE, mec="white", ms=7,
                  label="ensemble MSD"),
           Line2D([0], [0], color=RED, lw=2.2, label=r"fit  $\langle r^2\rangle = 4D\tau$")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
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
