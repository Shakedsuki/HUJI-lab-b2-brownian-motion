# -*- coding: utf-8 -*-
"""Single-row quantitative summary figure: (a) mean growth rate, (b) occupancy
phi, (c) effective fractal dimension D, all vs CuSO4 concentration on identical
x axes. Replaces the three separate column figures in the final report.
Data: radius_*.csv via report_figures.py loaders + fractalD_reliable.csv."""
import os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
sys.path.insert(0, r"C:/dev/brownian-motion/experiments/dla-concentration-final/scripts")
import report_figures as rf

BLUE = "#33658a"
RED = "#a4432f"

runs = [rf.load_run(r) for r in rf.RUNS]
concs = [r["conc"] for r in rf.RUNS]
lamp = [bool(r.get("lamp")) for r in rf.RUNS]
rates = [rf.growth_rate_umps(r) for r in runs]
phi = [rf.fill_fraction(r) for r in runs]
rows = [r for r in csv.DictReader(
    (l for l in open(os.path.join(HERE, "..", "data", "fractalD_reliable.csv"))
     if not l.startswith("#")))]
Dc = [float(r["conc"]) for r in rows]
Dv = [float(r["D_reliable"]) for r in rows]
De = [float(r["unc"]) for r in rows]

fig, (axG, axP, axD) = plt.subplots(1, 3, figsize=(9.6, 2.9))

def star(ax, xs, ys, flags):
    for x, y, f in zip(xs, ys, flags):
        if f:
            ax.plot(x, y, marker="*", ms=10, mfc="k", mec="w", mew=0.5,
                    ls="none", zorder=6)

gv = [g for g, e in rates]; ge = [e for g, e in rates]
axG.errorbar(concs, gv, yerr=ge, fmt="o-", color=RED, ms=4, capsize=2.5, lw=1)
star(axG, concs, gv, lamp)
axG.set_ylabel("mean growth rate\n$dR/dt$ [µm/s]", fontsize=8.5)
axG.set_title("(a)", fontsize=10, loc="left")

pv = [m for m, l, h in phi]
axP.errorbar(concs, pv, yerr=[[m - l for m, l, h in phi], [h - m for m, l, h in phi]],
             fmt="o-", color=BLUE, ms=4, capsize=2.5, lw=1)
star(axP, concs, pv, lamp)
axP.set_ylabel("occupancy  $\\phi = M/\\pi R^2$", fontsize=8.5)
axP.set_ylim(0, 0.75)
axP.set_title("(b)", fontsize=10, loc="left")

axD.axhline(1.71, ls="--", color="0.4", lw=1, label="2D DLA: 1.71")
axD.errorbar(Dc, Dv, yerr=De, fmt="o", color=BLUE, ms=4, capsize=2.5, lw=1)
lampD = [any(abs(c - cc) < 1e-9 and l for cc, l in zip(concs, lamp)) for c in Dc]
star(axD, Dc, Dv, lampD)
axD.set_ylabel("effective fractal dim.  $D$", fontsize=8.5)
axD.set_ylim(1.55, 2.05)
axD.legend(fontsize=7, loc="lower right", frameon=False)
axD.set_title("(c)", fontsize=10, loc="left")

for ax in (axG, axP, axD):
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25, lw=0.5)
fig.supxlabel("CuSO$_4$ concentration [%]", fontsize=9.5, y=0.03)
fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.21, wspace=0.32)
fig.savefig(os.path.join(OUT, "quant_vs_conc_row.png"), dpi=200)
fig.savefig(os.path.join(OUT, "quant_vs_conc_row.pdf"))
print("done")
