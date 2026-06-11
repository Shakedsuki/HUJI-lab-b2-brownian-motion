"""
kb_strip.py -- one-panel headline figure: per-bead k_B by run.

Every gate-passing particle's k_B,i / k_B^real is plotted as a jittered strip,
one column per run, ordered by nominal temperature; the bar is the run's median
with a 68% bootstrap CI (distribution-free -- the per-bead spread is non-Gaussian
and over-dispersed). Excluded runs (drift / convection / non-stationary) are shown
greyed for transparency. The horizontal line is the real k_B (exact by the SI
definition). Shows, in one view: every k_B measurement, the per-run medians, the
temperature trend, and the bead-to-bead over-dispersion (the chi^2>>1 result).

Real per-bead data via kb_grid.analyse_run (same gates/beads as the grid).
Writes figures/kb_strip.png (+ .pdf).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline import paths, physics
import kb_grid

KB = physics.K_B
RED, BLUE, GREY = "#c1272d", "#2b6cb0", "0.55"


def bootstrap_median_ci(x, nboot=5000, ci=68, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    meds = np.median(rng.choice(x, size=(nboot, len(x))), axis=1)
    lo, hi = np.percentile(meds, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.median(x)), float(lo), float(hi)


def load():
    mpp = paths.load_scale() or 0.14381
    rows = []
    for stem in sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:])):
        r = kb_grid.analyse_run(stem, mpp)
        if r is None or r["n"] < 3:
            continue
        for v in (r["fit"]["kb_i"].values / KB):
            rows.append(dict(run=stem, T=r["T"], ratio=float(v),
                             excluded=stem in kb_grid.EXCLUDED_RUNS, n=r["n"]))
    return pd.DataFrame(rows)


def main():
    df = load()
    runs = (df[["run", "T", "excluded", "n"]].drop_duplicates()
              .sort_values(["T", "run"], key=lambda c: c.map(
                  lambda v: int(v[3:]) if isinstance(v, str) else v))
              .reset_index(drop=True))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.axhline(1.0, color="0.3", lw=1.1, zorder=1)
    rng = np.random.default_rng(1)
    labels, edges, prevT = [], [], None

    for i, row in runs.iterrows():
        Trun = row["T"]
        beads = df.loc[df.run == row["run"], "ratio"].values
        col = GREY if row["excluded"] else RED
        jit = rng.uniform(-0.13, 0.13, len(beads))
        ax.plot(i + jit, beads, "o", ms=3.6, color=BLUE,
                alpha=0.30 if row["excluded"] else 0.7, mec="none", zorder=2)
        med, lo, hi = bootstrap_median_ci(beads)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="_", ms=18,
                    mew=2.6, color=col, ecolor=col, elinewidth=1.8, capsize=4,
                    zorder=3)
        lab = f"{row['run'].replace('run', '')}\n{Trun:g}°\n$n$={row['n']}"
        if row["excluded"]:
            lab += "\n(excl.)"
        labels.append(lab)
        if prevT is not None and Trun != prevT:
            edges.append(i - 0.5)
        prevT = Trun
    for e in edges:
        ax.axvline(e, color="0.9", lw=0.8, zorder=0)

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(r"$k_{B,i}\,/\,k_B^{\mathrm{real}}$", fontsize=12)
    ax.set_xlabel("run, ordered by nominal temperature", fontsize=11)
    ax.set_title(r"Per-bead $k_B$ by run   "
                 r"(bar: median $\pm$ 68% bootstrap CI;  line: real $k_B$, "
                 r"exact by SI definition)", fontsize=11)
    ax.set_ylim(0.2, 2.3)
    ax.margins(x=0.02)

    out = os.path.join(paths.FIGURES_DIR, "kb_strip.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    try:
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight"); pdf = " (+ .pdf)"
    except PermissionError:
        pdf = " (.pdf locked)"
    plt.close(fig)
    print(f"wrote {out}{pdf}; {runs['excluded'].eq(False).sum()} survivor + "
          f"{runs['excluded'].sum()} excluded runs")


if __name__ == "__main__":
    main()
