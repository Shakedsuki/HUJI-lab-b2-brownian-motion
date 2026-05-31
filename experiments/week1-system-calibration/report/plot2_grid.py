#!/usr/bin/env python3
"""
plot2_grid.py
=============
Report figure 2, multi-run grid -- the per-run Stokes-Einstein D-vs-1/r panels
(run3 / run4 / run6) side by side in one row, on shared axes so the slope (and
hence k_B) is directly comparable across runs.

Each panel reuses the clean plot2 machinery (robust D*r-outlier cut, per-bead
median through-origin fit). Axes are shared, so a steeper line literally means a
larger k_B. Per-panel k_B (as a fraction of the accepted value) is annotated.

Usage
-----
    python plot2_grid.py                         # run3, run4, run6
    python plot2_grid.py --runs run3 run4 run6 --T 25
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import plot2_pooled as pp
import plot2_D_vs_inv_r as p2


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["run3", "run4", "run6"])
    ap.add_argument("--T", type=float, default=25.0)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--delta-rho", type=float, default=50.0)
    ap.add_argument("--r-star", type=float, default=None)
    ap.add_argument("--exclude-hindered", action="store_true")
    ap.add_argument("--mad-k", type=float, default=3.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eta_cP = args.eta if args.eta is not None else p2.water_viscosity_cP(args.T)
    eta_Pa_s = eta_cP * 1e-3
    r_star = (args.r_star if args.r_star is not None
              else p2.sediment_r_star_um(args.T, args.delta_rho))

    # analyse each run, then use common axis limits so slopes are comparable
    results = [pp.analyze([r], args, eta_cP, eta_Pa_s, r_star, pooled=False)
               for r in args.runs]
    xmax = max(res["free"]["inv_r"].max() for res in results) * 1.08
    ymax = max(res["free"]["D_um2_s"].max() for res in results) * 1.12

    p2.set_style()
    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax, run, res in zip(axes[0], args.runs, results):
        pp.draw_panel(ax, res, args, full=False, xmax=xmax, ymax=ymax,
                      title=run, ylabel=(ax is axes[0, 0]))
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()

    tag = "-".join(r.replace("run", "") for r in args.runs)
    out = os.path.abspath(args.out or os.path.join(
        p2.ROOT, "figures", f"plot2_grid_runs{tag}.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved -> {out}\n")


if __name__ == "__main__":
    main()
