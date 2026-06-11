"""
kb_strip.py -- one-panel headline figure: per-bead k_B by run.

Every gate-passing particle's k_B,i / k_B^acc is plotted as a jittered strip,
one column per run, ordered by nominal temperature; the bar is the run's median
+/- its robust SE (1.4826*MAD/sqrt n -- the SAME error the grid and summary use,
so the bars are consistent across all figures). NB the bead strip itself shows
the (over-dispersed, non-Gaussian) spread; the bar shows the uncertainty on the
median. Excluded runs are greyed with their disqualifier. The horizontal line is
the accepted k_B (exact by the SI definition). One view: every k_B measurement,
the per-run medians, the temperature trend, and the bead-to-bead over-dispersion
(the chi^2>>1 result).

[A bootstrap median CI was tried (per the mock) but rejected: for these small n
the central beads cluster tightly so it collapses to a razor line that misreads
as precision -- e.g. run15 beads 0.32,0.33,0.88,0.91,0.92,1.07,1.16 give a
0.88-0.92 CI although the run is plainly over-dispersed. The MAD-SE is honest and
consistent.]

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
# short on-figure disqualifier per excluded run (full text in kb_grid.EXCLUDED_RUNS)
EXCL_REASON = {"run6": "convection", "run10": "drift", "run11": "drift",
               "run12": "convection", "run13": "heated", "run14": "non-stat."}


def load():
    mpp = paths.load_scale() or 0.14381
    beads, runs = [], []
    for stem in sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:])):
        r = kb_grid.analyse_run(stem, mpp)
        if r is None or r["n"] < 3:
            continue
        exc = stem in kb_grid.EXCLUDED_RUNS
        runs.append(dict(run=stem, T=r["T"], n=r["n"], excluded=exc,
                         med=r["kb_med"] / KB, se=r["se_kb_med"] / KB,
                         reason=EXCL_REASON.get(stem, "")))
        for v in (r["fit"]["kb_i"].values / KB):
            beads.append(dict(run=stem, T=r["T"], ratio=float(v)))
    return pd.DataFrame(beads), pd.DataFrame(runs)


def main():
    beadsdf, runs = load()
    runs = runs.sort_values(["T", "run"], key=lambda c: c.map(
        lambda v: int(v[3:]) if isinstance(v, str) else v)).reset_index(drop=True)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.axhline(1.0, color="0.3", lw=1.1, zorder=1)
    rng = np.random.default_rng(1)
    labels, edges, prevT = [], [], None

    for i, row in runs.iterrows():
        Trun = row["T"]
        beads = beadsdf.loc[beadsdf.run == row["run"], "ratio"].values
        col = GREY if row["excluded"] else RED
        jit = rng.uniform(-0.13, 0.13, len(beads))
        ax.plot(i + jit, beads, "o", ms=3.6, color=BLUE,
                alpha=0.30 if row["excluded"] else 0.7, mec="none", zorder=2)
        ax.errorbar(i, row["med"], yerr=row["se"], fmt="_", ms=18, mew=2.6,
                    color=col, ecolor=col, elinewidth=1.8, capsize=4, zorder=3)
        lab = f"{row['run'].replace('run', '')}\n{Trun:g}°\n$n$={row['n']}"
        if row["excluded"]:
            lab += f"\n(excl.:\n{row['reason']})"
        labels.append(lab)
        if prevT is not None and Trun != prevT:
            edges.append(i - 0.5)
        prevT = Trun
    for e in edges:
        ax.axvline(e, color="0.9", lw=0.8, zorder=0)

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(r"$k_{B,i}\,/\,k_B^{\mathrm{acc}}$", fontsize=12)
    ax.set_xlabel("run, ordered by nominal temperature", fontsize=11)
    ax.set_title(r"Per-bead $k_B$ by run   "
                 r"(bar: median $\pm$ SE;  line: accepted $k_B$, "
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
