#!/usr/bin/env python3
"""
plot3_kb_strip.py
=================
Report figure 3 -- per-bead k_B "strip" for the room-temperature runs (the
week-1 companion of week-2's kb_strip).

Where plot2 fits a line to D-vs-(1/r), this figure shows the individual k_B that
each free bead implies,

        k_B,i = 6 pi eta(T) r_i D_i / T ,

one jittered point per bead, grouped by run, with the per-run median +/- SE drawn
as a crossbar and the accepted k_B as a reference line. It makes the run-to-run
reproducibility (and the bead-to-bead scatter behind every plot2 slope) visible
at a glance.

Policy (week-2 parity)
----------------------
WALL-PINNED beads are EXCLUDED from the headline. The polyethylene spheres are
buoyant, so beads larger than the sedimentation scale r*(T) graze the top
coverslip and sub-diffuse (D below the free-diffusion line), which biases k_B
low. They are shown as open grey markers (visible, not hidden) but do not enter
the per-run or pooled medians -- exactly as week-2 removes them. Gross mislinks
are dropped first by a robust MAD cut on D*r (Stokes-Einstein => D*r ~ const).

The error bar is the STATISTICAL SE of the median (1.2533 * 1.4826 * MAD / sqrt n);
the common-mode radius / temperature systematics are shared by all runs and are
NOT shown here (they shift every point together) -- see plot2 for those.

Reuses plot2_D_vs_inv_r (loaders, physics) so the bead set and k_B definition are
identical to the D-vs-(1/r) figures.

Usage
-----
    python plot3_kb_strip.py                          # run3,4,5,6 ; T=25 C
    python plot3_kb_strip.py --runs run3 run4 --T 25
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import plot2_D_vs_inv_r as p2     # loaders, physics, constants


RUN_COLORS = {
    "run3": "#1f77b4", "run4": "#2ca02c", "run5": "#ff7f0e", "run6": "#9467bd",
    "run7": "#17becf", "run8": "#bcbd22", "run9": "#e377c2", "run10": "#8c564b",
}
ACCENT = "#c1272d"


def median_se(x):
    """Median + robust SE of the median (1.2533 * 1.4826 * MAD / sqrt n)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    m = float(np.median(x))
    mad = float(np.median(np.abs(x - m)))
    return m, 1.2533 * 1.4826 * mad / np.sqrt(len(x))


def per_run(run, T, eta_Pa_s, r_star, mad_k):
    """Per-bead k_B/k_B^acc for one run: free (headline) + wall-hindered (shown).

    Returns dict(run, free_ratio, wall_ratio, med, se, n_free, n_wall).
    """
    df = p2.load_beads(run)
    # robust D*r mislink cut (Stokes-Einstein => D*r ~ const)
    k = (df["D_um2_s"] * df["r_um"]).values
    med = np.median(k)
    mad = np.median(np.abs(k - med))
    if mad > 0:
        df = df[np.abs(k - med) <= mad_k * mad].copy()

    free = df[df["r_um"] <= r_star]
    wall = df[df["r_um"] > r_star]

    def ratio(g):
        return (p2.kB_per_bead(g["D_um2_s"].values, g["r_um"].values, T, eta_Pa_s)
                / p2.K_B_ACCEPTED)

    free_ratio = ratio(free)
    wall_ratio = ratio(wall)
    m, se = median_se(free_ratio)
    return dict(run=run, free_ratio=free_ratio, wall_ratio=wall_ratio,
                med=m, se=se, n_free=len(free), n_wall=len(wall))


def jitter(n, width=0.16):
    """Deterministic symmetric x-jitter for n points (no RNG -> reproducible)."""
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["run3", "run4", "run5", "run6"])
    ap.add_argument("--T", type=float, default=25.0, help="temperature [C]")
    ap.add_argument("--eta", type=float, default=None,
                    help="viscosity [cP] (default: water at T)")
    ap.add_argument("--delta-rho", type=float, default=50.0,
                    help="|bead-fluid| density mismatch [kg/m^3] for r*")
    ap.add_argument("--r-star", type=float, default=None,
                    help="override the free-diffusion radius cut [um]")
    ap.add_argument("--mad-k", type=float, default=3.5,
                    help="robust D*r outlier cut in MAD units (mislink removal)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eta_cP = args.eta if args.eta is not None else p2.water_viscosity_cP(args.T)
    eta_Pa_s = eta_cP * 1e-3
    r_star = (args.r_star if args.r_star is not None
              else p2.sediment_r_star_um(args.T, args.delta_rho))

    results = [per_run(r, args.T, eta_Pa_s, r_star, args.mad_k)
               for r in args.runs]

    pooled = np.concatenate([r["free_ratio"] for r in results])
    pool_m, pool_se = median_se(pooled)

    print(f"\nWeek-1 k_B strip: T={args.T:.1f}C  eta={eta_cP:.3f}cP  "
          f"r*={r_star:.2f}um  (wall-pinned r>r* excluded)")
    for r in results:
        print(f"  {r['run']}: n_free={r['n_free']} (+{r['n_wall']} wall)  "
              f"k_B={r['med']:.2f} +/- {r['se']:.2f} x")
    print(f"  POOLED free: n={len(pooled)}  "
          f"k_B={pool_m:.2f} +/- {pool_se:.2f} x accepted")

    p2.set_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # accepted k_B reference + pooled free median band
    ax.axhline(1.0, color="0.35", lw=1.2, zorder=2,
               label=r"accepted $k_B$ (exact, SI definition)")
    ax.axhspan(pool_m - pool_se, pool_m + pool_se, color=ACCENT, alpha=0.10,
               zorder=0)
    ax.axhline(pool_m, color=ACCENT, ls="--", lw=1.4, zorder=2,
               label=rf"pooled free median = {pool_m:.2f}$\pm${pool_se:.2f}$\,"
                     rf"k_B^{{\rm acc}}$ ($n={len(pooled)}$)")

    for i, r in enumerate(results):
        col = RUN_COLORS.get(r["run"], "#444444")
        # free beads (headline) -- filled, coloured by run
        yr = r["free_ratio"]
        ax.plot(i + jitter(len(yr)), yr, "o", ms=4.5, color=col, mec="white",
                mew=0.4, alpha=0.85, zorder=3)
        # wall-hindered beads -- open grey, excluded from the median
        yw = r["wall_ratio"]
        if len(yw):
            ax.plot(i + jitter(len(yw)), yw, "o", ms=4.5, mfc="none",
                    mec="0.6", mew=1.0, alpha=0.8, zorder=3)
        # median +/- SE crossbar
        ax.errorbar(i, r["med"], yerr=r["se"], fmt="none", ecolor=ACCENT,
                    elinewidth=2.0, capsize=5, capthick=2.0, zorder=5)
        ax.plot([i - 0.22, i + 0.22], [r["med"], r["med"]], "-", color=ACCENT,
                lw=2.6, zorder=6)

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"{r['run']}\n$n$={r['n_free']}"
                        + (f" (+{r['n_wall']} wall)" if r["n_wall"] else "")
                        for r in results])
    ax.set_xlim(-0.6, len(results) - 0.4)
    ax.set_ylabel(r"$k_{B,i}\,/\,k_B^{\mathrm{acc}}$")
    ax.set_xlabel("run  (all at room temperature)")
    ax.set_title("Week-1 per-bead $k_B$ by run "
                 r"($k_{B,i}=6\pi\eta(T)\,r_i D_i / T$; "
                 r"free spheres $r\leq r^*$)", fontsize=11.5)

    from matplotlib.lines import Line2D
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([0], [0], marker="o", color="w", mfc="#1f77b4", mec="white",
               ms=7, label=r"free sphere ($r\leq r^*$); bar = median $\pm$ SE"),
        Line2D([0], [0], marker="o", color="w", mfc="none", mec="0.6", ms=7,
               label=rf"wall-pinned ($r>r^*\approx{r_star:.2f}\,\mu$m), excluded")]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, ncol=1)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.005,
             rf"$T={args.T:.0f}\,^\circ$C, $\eta={eta_cP:.3f}$ cP.  Bars are the "
             r"STATISTICAL SE only; common-mode radius ($\pm$1 px) and $T$ "
             r"systematics shift all points together (see plot2).",
             ha="center", va="bottom", fontsize=8.5, color="0.4")

    tag = "-".join(r.replace("run", "") for r in args.runs)
    out = os.path.abspath(args.out or os.path.join(
        p2.ROOT, "figures", f"kb_strip_runs{tag}.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    try:
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
    except PermissionError:
        pass
    print(f"\nsaved -> {out}\n")


if __name__ == "__main__":
    main()
