"""
kb_grid_pub.py -- clean, publication-ready per-measurement k_B grid.

One panel per run that PASSED the gates (10 of 16); within each panel only the
gate-passing beads are shown. Each panel is a Stokes-Einstein D-vs-(1/r) plot
whose slope is the measured k_B (per-bead median). Minimal ink: points, the
measured-k_B line with its statistical band, a faint accepted-k_B reference, and
one line of text -- run, temperature, and the measured k_B. No gate funnels, no
chi2/eta/n boxes, no cross-check line, no drift markers (those live in kb_grid.py).

Reuses kb_grid.analyse_run, so the numbers are identical to the working grid.
Writes figures/kb_grid_publication.png (+ .pdf).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from pipeline import paths, physics
import kb_grid

KB = physics.K_B


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

    xmax = max((1 / r["fit"]["r_um"]).max() for r in res) * 1.06
    ymax = max(r["fit"]["D_um2_s"].max() for r in res) * 1.10

    plt.rcParams.update({
        "font.size": 11, "axes.linewidth": 0.8, "xtick.direction": "out",
        "ytick.direction": "out", "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    })
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 6.3),
                             sharex=True, sharey=True)
    BLUE, RED = "#2b6cb0", "#c1272d"
    xs = np.array([0.0, xmax])

    for ax, r in zip(axes.flat, res):
        f = r["fit"]
        # faint accepted-k_B reference (slope from accepted k_B at this T)
        pref = physics.kB_prefactor(r["T"]) * 1e-18
        ax.plot(xs, (KB / pref) * xs, "--", color="0.7", lw=1.1, zorder=1)
        # measured k_B = per-bead median slope, with +/- statistical band
        sm = r["slope_med"]; se = r["se_kb_med"] / pref
        ax.fill_between(xs, (sm - se) * xs, (sm + se) * xs, color=RED,
                        alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, sm * xs, "-", color=RED, lw=2.0, zorder=4)
        # gate-passing beads
        ax.plot(f["inv_r"], f["D_um2_s"], "o", ms=4.5, mfc=BLUE, mec="white",
                mew=0.5, zorder=5)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=9, length=3)
        ax.grid(True, color="0.92", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        # one minimal text block: run . T . measured k_B
        ax.text(0.05, 0.95, f"{r['run']}  ·  {r['T']:.1f} °C",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                color="0.35")
        ax.text(0.05, 0.85, rf"$k_B = {r['kb_med']/KB:.2f}\,k_B^{{\rm real}}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=12,
                color=RED)
        ax.text(0.05, 0.74, rf"$n = {r['n']}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.5, color="0.45")

    for i in range(nrows):
        axes[i][0].set_ylabel(r"$D$   [$\mu$m$^2$ s$^{-1}$]", fontsize=11)
    for j in range(ncols):
        axes[nrows - 1][j].set_xlabel(r"$1/r$   [$\mu$m$^{-1}$]", fontsize=11)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", mfc=BLUE, mec="white", ms=8,
                  label="particles"),
           Line2D([0], [0], color=RED, lw=2.2, label=r"measured $k_B$ (per-bead median)"),
           Line2D([0], [0], color="0.7", lw=1.4, ls="--", label=r"real $k_B$ value")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
    out = os.path.join(paths.FIGURES_DIR, "kb_grid_publication.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    try:
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
        pdf = " (+ .pdf)"
    except PermissionError:
        pdf = " (.pdf locked -- close the viewer and rerun for the vector copy)"
    plt.close(fig)
    print(f"wrote {out}{pdf}; {len(res)} gate-passing runs")


if __name__ == "__main__":
    main()
