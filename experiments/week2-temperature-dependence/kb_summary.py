"""
kb_summary.py -- Week-2 SYNTHESIS figure (publication): k_B/k_B^acc vs nominal T.

One point per run = the robust per-bead MEDIAN k_B -- NOT the through-origin
slope. With the narrow ~0.8-1.4 um^-1 lever arm the slope is estimator-unstable
(run7 is 1.00x by slope, 1.28x by median); the median never regresses on 1/r, so
it is the headline and the slope is demoted to the per-run grid cross-check.

Per-run bead set: hand-tagged free diffusers (r <= r*(T)), drift-excluded
(|v|>0.1 um/s AND >2 sigma_v), alpha in [0.7,1.3] (free-diffusion validity),
D-quality gate. Error bars = per-bead median SE (1.4826*MAD/sqrt n) -- the
STATISTICAL error only.

Common-mode SYSTEMATICS (do NOT average down) shown/stated separately:
  * radius offset: hand tags carry a roughly constant diffraction over-read.
    The median is immune to radius SCATTER but fully exposed to a common-mode
    OFFSET: +/-1 px on r~1 um beads = +/-15% on k_B. Drawn as the shaded band.
  * temperature label: +/-1 C -> ~3% via eta(T) (and the trend below says the
    real T-label error is far larger than +/-1 C).

Model: if the fluid never reached the stage setpoint but sat at ambient T_amb,
then assigning eta(T_nom), T_nom in k_B = 6 pi eta r D / T gives
    k_B,ext / k_B,acc = [eta(T_nom)/eta(T_true)] * (T_true_K / T_nom_K),
with T_true = T_nom + f (T_amb - T_nom);  f in [0,1] = thermal-decoupling
fraction (0 = perfect control, 1 = fully ambient). f is FIT to the SURVIVING
runs only, so contaminated/excluded runs cannot drag the one parameter.

!!! T_AMB MUST BE THE LOGGED ROOM TEMPERATURE. The 20 C default is a PLACEHOLDER;
    with it the model is asserted, not validated -- replace before submission. !!!

Excluded runs are greyed with their reason (excluded-but-shown survives a
defense; silently absent does not). Reproducible: reads measurements/, writes
figures/kb_summary.png + .csv.
"""
import os
import numpy as np
import pandas as pd

from pipeline import paths, physics, figstyle

KB = physics.K_B
MPP = paths.load_scale() or 0.14381
T_AMB = 20.0                       # !!! PLACEHOLDER -- use the logged room temp
RADIUS_OFFSET_PX = 1.0             # common-mode radius systematic (+/- this)

# run -> nominal T; classification with reasons (None = headline survivor)
RUNS = {
    "run2": 14.0, "run3": 14.0, "run4": 14.0, "run5": 15.2, "run6": 15.2,
    "run7": 16.8, "run8": 16.8, "run9": 20.0, "run10": 20.0, "run11": 20.0,
    "run12": 23.78, "run13": 24.3, "run14": 24.3, "run15": 30.3, "run16": 30.3,
}
EXCLUDED = {
    "run6":  "convection (5/8 drift)",
    "run10": "residual drift",
    "run11": "residual drift",
    "run12": "convection",
    "run14": "non-stationary (D(t) ramp)",
}


def per_run(stem, T):
    """Median k_B/KB and per-bead median SE over the headline bead set."""
    out = paths.out_dir(stem, make=False)
    fman = os.path.join(out, "radius_manual.csv")
    fm = os.path.join(out, "msd.csv")
    if not (os.path.exists(fman) and os.path.exists(fm)):
        return None
    man = pd.read_csv(fman)[["particle", "r_um_manual"]]
    d = pd.read_csv(fm).merge(man, on="particle", how="inner")
    rstar = physics.sediment_r_star_um(T)
    f = d[(~d.get("drift_flag", False).fillna(False))
          & (d["r_um_manual"] <= rstar)
          & (d["alpha"].between(0.7, 1.3))
          & ((d["D_err"] / d["D_um2_s"]) < 0.5)].copy()
    if len(f) < 3:
        return None
    kb = physics.kB_per_bead(f["D_um2_s"].values, f["r_um_manual"].values, T) / KB
    m = float(np.median(kb))
    se = float(np.median(np.abs(kb - m)) * 1.4826 / np.sqrt(len(kb)))
    return dict(run=stem, T=T, ratio=m, se=se, n=len(kb))


def model_ratio(T_nom, f):
    """k_B,ext/k_B,acc if the sample sits at T_true = T_nom + f(T_amb-T_nom)."""
    T_nom = np.asarray(T_nom, float)
    T_true = T_nom + f * (T_AMB - T_nom)
    eta_n = physics.water_viscosity_cP(T_nom)
    eta_t = np.array([physics.water_viscosity_cP(t) for t in np.atleast_1d(T_true)])
    return (eta_n / eta_t) * ((T_true + 273.15) / (T_nom + 273.15))


def fit_f(rows):
    """LS fit of the decoupling fraction f to survivor (T, ratio, se)."""
    T = np.array([r["T"] for r in rows]); y = np.array([r["ratio"] for r in rows])
    s = np.array([r["se"] for r in rows])
    fs = np.linspace(0.0, 1.5, 1501)
    chi = np.array([np.sum(((y - model_ratio(T, f)) / s) ** 2) for f in fs])
    i = int(np.argmin(chi))
    fhat, chimin = fs[i], chi[i]
    # 1-sigma from delta-chi2 = 1
    lo = fs[:i][chi[:i] <= chimin + 1]; hi = fs[i:][chi[i:] <= chimin + 1]
    sig = (((hi[-1] if len(hi) else fhat) - (lo[0] if len(lo) else fhat)) / 2) or np.nan
    return fhat, sig, chimin / max(len(rows) - 1, 1)


def main():
    rows = [per_run(s, T) for s, T in RUNS.items()]
    rows = [r for r in rows if r]
    surv = [r for r in rows if r["run"] not in EXCLUDED]
    excl = [r for r in rows if r["run"] in EXCLUDED]
    fhat, sig_f, chi2dof = fit_f(surv)

    # pooled headline (all survivor beads) for the caption
    allkb = []
    for r in surv:
        d = pd.read_csv(os.path.join(paths.out_dir(r["run"], make=False), "msd.csv")).merge(
            pd.read_csv(os.path.join(paths.out_dir(r["run"], make=False),
                                     "radius_manual.csv"))[["particle", "r_um_manual"]],
            on="particle")
        rstar = physics.sediment_r_star_um(r["T"])
        f = d[(~d.get("drift_flag", False).fillna(False)) & (d["r_um_manual"] <= rstar)
              & (d["alpha"].between(0.7, 1.3)) & ((d["D_err"] / d["D_um2_s"]) < 0.5)]
        allkb += list(physics.kB_per_bead(f["D_um2_s"].values, f["r_um_manual"].values, r["T"]) / KB)
    pooled = float(np.median(allkb))

    pd.DataFrame(rows).to_csv(os.path.join(paths.FIGURES_DIR, "kb_summary.csv"), index=False)
    print(f"survivors fit: f = {fhat:.2f} +/- {sig_f:.2f}  (0=control,1=ambient), "
          f"chi2/dof = {chi2dof:.1f}; pooled median = {pooled:.2f}x over n={len(allkb)}")

    import matplotlib.pyplot as plt
    figstyle.set_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    Tg = np.linspace(12.5, 32, 240)

    # common-mode radius-offset systematic band around the fitted model
    base = model_ratio(Tg, fhat)
    # +/-1px offset on r~1um: multiplicative ~ (rbar +/- dpx*mpp)/rbar
    rbar = np.median([r["ratio"] for r in surv]) * 0 + 1.0  # ~1 um typical bead
    off = RADIUS_OFFSET_PX * MPP / 1.0
    ax.fill_between(Tg, base * (1 - off), base * (1 + off), color="#1f77b4",
                    alpha=0.12, zorder=1,
                    label=rf"$\pm${RADIUS_OFFSET_PX:.0f} px common-mode radius offset ($\pm${off*100:.0f}%)")
    ax.axhline(1.0, color="0.4", lw=1.0, zorder=1)

    # fitted model + full-ambient (f=1) reference
    ax.plot(Tg, base, "--", color="#1f77b4", lw=1.8, zorder=3,
            label=rf"sample near ambient: $f$={fhat:.2f}$\pm${sig_f:.2f}"
                  rf" ($T_{{\rm amb}}$={T_AMB:.0f} °C),  $\chi^2/$dof={chi2dof:.1f}")
    ax.plot(Tg, model_ratio(Tg, 1.0), ":", color="0.55", lw=1.3, zorder=2,
            label="full decoupling ($f$=1) reference")

    # x-jitter so same-T runs (survivors + excluded together) don't overlap
    from collections import defaultdict
    byT = defaultdict(list)
    for r in rows:
        byT[r["T"]].append(r["run"])
    jit = {}
    for T, runs_ in byT.items():
        order = sorted(runs_, key=lambda s: int(s[3:]))
        for i, run in enumerate(order):
            jit[run] = (i - (len(order) - 1) / 2) * 0.34

    def draw(rows_, **kw):
        for r in rows_:
            x = r["T"] + jit[r["run"]]
            ax.errorbar(x, r["ratio"], yerr=r["se"], fmt="o", ms=6,
                        capsize=3, zorder=5, **kw)
            ax.annotate(r["run"].replace("run", ""), (x, r["ratio"]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=7.5, color="0.4")
    draw(surv, color="#c1272d", ecolor="#c1272d", elinewidth=1.2, mec="white", mew=0.5)
    draw(excl, color="0.6", ecolor="0.6", elinewidth=1.0, mfc="none", mec="0.6")

    from matplotlib.lines import Line2D
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c1272d",
               markersize=7, label=r"per-run median $k_B$ (in fit)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="0.6", markersize=7,
               label="excluded (drift / convection / non-stationary)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left", ncol=1)

    ax.set_xlabel("nominal sample temperature  [°C]")
    ax.set_ylabel(r"$k_B \,/\, k_B^{\mathrm{acc}}$")
    ax.set_title("Week-2: extracted $k_B$ falls with nominal $T$ "
                 "— the sample sat near ambient\n"
                 r"(per-run median estimator; bars = statistical SE; "
                 r"labels = run id)", fontsize=11)
    ax.set_xlim(12.5, 32); ax.set_ylim(0.5, 1.55)
    p = figstyle.save(fig, os.path.join(paths.FIGURES_DIR, "kb_summary.png"))
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
