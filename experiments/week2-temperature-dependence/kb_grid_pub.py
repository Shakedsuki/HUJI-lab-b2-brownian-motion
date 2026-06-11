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
    ncols = 3 if len(res) <= 9 else 5
    nrows = int(np.ceil(len(res) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(res):]:
        ax.set_visible(False)
    BLUE, RED = "#2b6cb0", "#c1272d"
    xs = np.array([0.0, xmax])

    for ax, r in zip(axes.flat, res):
        f = r["fit"]
        pref = physics.kB_prefactor(r["T"]) * 1e-18
        # measured k_B = per-bead median slope + its +/- statistical SE band
        sm = r["slope_med"]; se = r["se_kb_med"] / pref
        ax.fill_between(xs, (sm - se) * xs, (sm + se) * xs, color=RED,
                        alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, sm * xs, "-", color=RED, lw=2.0, zorder=4)
        # drift/alpha-gated beads (NOT in the fit): faint open circles -> n self-documents
        dr = r["df"][r["df"]["drift_flag"]]
        if len(dr):
            ax.plot(1 / dr["r_um"], dr["D_um2_s"], "o", ms=4.5, mfc="none",
                    mec="0.6", mew=0.9, zorder=3)
        # gate-passing beads with sigma_D (vert) + sigma_1/r (horiz), faded
        ax.errorbar(f["inv_r"], f["D_um2_s"], xerr=f["sig_invr"], yerr=f["D_err"],
                    fmt="o", ms=4.5, mfc=BLUE, mec="white", mew=0.5,
                    ecolor="0.7", elinewidth=0.7, capsize=0, zorder=5)
        # accepted-k_B reference -- drawn ON TOP so it stays visible where it
        # nearly coincides with the measurement (e.g. run3)
        ax.plot(xs, (KB / pref) * xs, color="0.45", lw=1.2, ls=(0, (6, 3)), zorder=6)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=9, length=3)
        ax.grid(True, color="0.92", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.05, 0.95, f"{r['run']}  ·  {r['T']:.1f} °C",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                color="0.35")
        ax.text(0.05, 0.85,
                rf"$k_B = {r['kb_med']/KB:.2f}\pm{r['se_kb_med']/KB:.2f}\,k_B^{{\rm acc}}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=11, color=RED)
        ax.text(0.05, 0.74, rf"$n = {r['n']}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.5, color="0.45")

    for i in range(nrows):
        axes[i][0].set_ylabel(r"$D$   [$\mu$m$^2$ s$^{-1}$]", fontsize=11)
    for j in range(ncols):
        axes[nrows - 1][j].set_xlabel(r"$1/r$   [$\mu$m$^{-1}$]", fontsize=11)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", mfc=BLUE, mec="white", ms=8,
                  label=r"particles (bars: $\sigma_D,\ \sigma_{1/r}$)"),
           Line2D([0], [0], marker="o", color="w", mfc="none", mec="0.6", mew=1.2,
                  ms=8, label="cut by drift/$\\alpha$ gates (excl.)"),
           Line2D([0], [0], color=RED, lw=2.2, label=r"measured $k_B$ (per-bead median)"),
           Line2D([0], [0], color="0.45", lw=1.4, ls=(0, (6, 3)), label=r"accepted $k_B$")]
    fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.012))
    fig.text(0.5, -0.05, r"Red band = $\pm1\sigma$ statistical SE on the median "
             r"slope (= the $\pm$ on $k_B$).  $n$ = particles passing the drift "
             r"and $\alpha\in[0.7,1.3]$ gates (filled); gated-out beads shown open.",
             ha="center", va="top", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
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
