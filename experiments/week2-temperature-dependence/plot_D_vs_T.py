"""
plot_D_vs_T.py -- Week-2 figure: diffusion coefficient vs temperature.

The direct, radius-free view of the temperature sweep behind the k_B result.
Beads differ in radius and D = k_B T / (6 pi eta(T) r), so a bare D mixes size
with temperature. We therefore plot D SCALED TO A FIXED radius r_ref = 1 um,

        D@1um,i = D_i * (r_i / r_ref)        [um^2/s, r_ref = 1 um]

which is just the radius-free Stokes-Einstein group D*r (= k_B T / 6 pi eta)
expressed as the diffusion coefficient a 1-um-radius bead would have. The figure
keeps the report's clean look -- no title, minimal axes (D [um^2/s] vs
temperature), trimmed two-entry legend -- but shows the supporting detail: the
faint per-bead cloud (jittered by run), the per-T median (robust MAD-SE on y,
+/-1 C T-label on x) with its n annotated, and the parameter-free Stokes-Einstein
curve, on a light grid. Numbers also in D_vs_T_summary.csv.

The only curve is the PARAMETER-FREE Stokes-Einstein prediction at the accepted
k_B,

        D@1um(T) = k_B^acc * T / (6 pi eta(T) * r_ref) ,

NOT a fit. If the sample were at its nominal T the medians would sit on it. They
do not: the measured D is nearly flat while the prediction climbs ~55 % across
14 -> 30 C, because the cell never reached the stage setpoint and sat near
ambient (the same story quantified in kb_summary.py). The deviation from this
green line at each T IS the k_B/k_B^acc ratio of kb_grid / kb_summary.

Bead set, gates and run exclusions are delegated to kb_grid.analyse_run, so the
points here are IDENTICAL to the k_B grid's measurement set (hand-tagged radii,
free diffusers r <= r*, alpha in [0.7,1.3], drift-excluded). Reproducible: reads
measurements/, writes figures/D_vs_T.png (+ .pdf) and figures/D_vs_T_summary.csv.
"""
import os

import numpy as np
import pandas as pd

from pipeline import paths, physics, figstyle
import kb_grid

KB = physics.K_B
R_REF_UM = 1.0          # reference radius for the scaled D [um]
BLUE, GREEN, GREY = "#2b6cb0", "#2ca02c", "#9aa0a6"


def se_curve_um2_s(T_C):
    """Stokes-Einstein D at r = R_REF_UM and accepted k_B [um^2/s].

    D = k_B T / (6 pi eta r);  with r = R_REF_UM this equals the radius-free
    group (k_B T / 6 pi eta) divided by r_ref, i.e. *1e18 / R_REF_UM in um^2/s.
    """
    eta = physics.water_viscosity_Pa_s(T_C)
    return KB * (T_C + 273.15) / (6.0 * np.pi * eta) * 1e18 / R_REF_UM


def main():
    mpp = paths.load_scale() or 0.14381
    allruns = sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:]))

    rows = []           # per (run) free-bead frame, tagged with T and run
    for stem in allruns:
        if stem in kb_grid.EXCLUDED_RUNS:
            continue
        res = kb_grid.analyse_run(stem, mpp)
        if res is None or res["n"] < 3:
            continue
        fit = res["fit"].copy()
        # D scaled to r_ref = 1 um (= D*r in um^2/s for r_ref = 1 um)
        fit["D_scaled"] = fit["D_um2_s"] * fit["r_um"] / R_REF_UM
        fit["T"] = res["T"]
        fit["T_unc"] = res["T_unc"]
        fit["run"] = stem
        rows.append(fit[["run", "T", "T_unc", "D_scaled"]])
    if not rows:
        raise SystemExit("no analysed runs found -- run process_all.py first")
    allfree = pd.concat(rows, ignore_index=True)

    # per-temperature median + robust SE (same estimator as kb_grid)
    tab = []
    for T, gp in allfree.groupby("T"):
        m, se = kb_grid.median_se(gp["D_scaled"].values)
        pred = float(se_curve_um2_s(T))
        tab.append(dict(T=float(T), T_unc=float(gp["T_unc"].iloc[0]),
                        n_free=len(gp), n_runs=gp["run"].nunique(),
                        eta_cP=round(physics.water_viscosity_cP(float(T)), 3),
                        D_at_1um=m, D_at_1um_se=se, D_SE_pred=pred,
                        ratio=m / pred))
    tab = pd.DataFrame(tab).sort_values("T").reset_index(drop=True)
    tab.to_csv(os.path.join(paths.FIGURES_DIR, "D_vs_T_summary.csv"), index=False)

    print(f"Week-2 D vs T  (D scaled to r={R_REF_UM:.0f} um; free, drift-excluded)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    import matplotlib.pyplot as plt
    figstyle.set_style()
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # parameter-free Stokes-Einstein prediction at accepted k_B
    Tg = np.linspace(allfree["T"].min() - 2, allfree["T"].max() + 2, 240)
    ax.plot(Tg, se_curve_um2_s(Tg), "-", color=GREEN, lw=2.4, zorder=3,
            label=r"Stokes-Einstein ($k_B^{\mathrm{acc}}$)")

    # faint per-bead points, x-jittered by run so same-T runs separate
    runs = sorted(allfree["run"].unique(), key=lambda s: int(s[3:]))
    for i, run in enumerate(runs):
        gp = allfree[allfree["run"] == run]
        jit = (i - (len(runs) - 1) / 2) * 0.12
        ax.scatter(gp["T"] + jit, gp["D_scaled"], s=16, color=GREY, alpha=0.45,
                   edgecolors="none", zorder=2)

    # per-T median, with +/-SE (y) and +/-1 C (x) bars
    ax.errorbar(tab["T"], tab["D_at_1um"], yerr=tab["D_at_1um_se"],
                xerr=tab["T_unc"], fmt="o", ms=8, color=BLUE, mec="white",
                mew=0.8, ecolor=BLUE, elinewidth=1.6, capsize=4, zorder=5,
                label=r"per-$T$ median $D$")
    for _, r in tab.iterrows():
        ax.annotate(f"n={r['n_free']:.0f}", (r["T"], r["D_at_1um"]),
                    textcoords="offset points", xytext=(8, 7),
                    fontsize=9, color="0.4")

    ax.set_xlabel(r"temperature  [$^\circ$C]")
    ax.set_ylabel(r"$D$  [$\mu$m$^2$/s]")
    ax.set_ylim(0, None)
    ax.margins(x=0.05)
    ax.legend(loc="upper left", fontsize=10, frameon=False, handlelength=1.6)

    out = os.path.join(paths.FIGURES_DIR, "D_vs_T.png")
    p = figstyle.save(fig, out, dpi=300)
    try:
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight"); pdf = " (+ .pdf)"
    except PermissionError:
        pdf = " (.pdf locked)"
    plt.close(fig)
    print(f"\nwrote {p}{pdf} and D_vs_T_summary.csv")


if __name__ == "__main__":
    main()
