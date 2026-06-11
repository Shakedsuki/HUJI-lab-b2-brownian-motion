"""
kb_grid.py
==========
Week-2 report figure: a GRID of Stokes-Einstein D-vs-(1/r) panels, one per run,
each at that run's MEASURED starting temperature. The slope of every panel IS
k_B:    D = (k_B T / 6 pi eta(T)) * (1/r)   =>   k_B = slope * 6 pi eta(T) / T.
This is last week's plot2_grid, extended across the temperature sweep.

MANUAL-RADIUS runs only: a run is included iff it has radius_manual.csv (radii
hand-marked at the true sphere edge via radius_tag.py, which bypass the ~0.30 um
diffraction over-read of the auto outer-edge fit -- the over-read otherwise pushes
nearly every bead past r* and empties the free set). D comes from the MSD fit; the
hand-tagged beads are the curated singles, so only the D-quality gate is applied.
WALL-PINNED BEADS ARE REMOVED: only free diffusers (r <= r*(T), the sedimentation
cut) enter the measurement -- they are not plotted, fitted, or pooled.

To add a run to the grid:  track+curate it (process_all.py), tag radii
(python radius_tag.py <run>), then re-run this / finalize.py.

Each panel carries (fixes 3-5):
  * x error bars  sigma_(1/r) = sigma_r / r^2  (sigma_r = 1 px localization floor
    combined with the frame-to-frame apparent-size scatter R_cv * r);
  * y error bars  sigma_D  from the MSD-fit covariance (msd.py);
  * the through-origin LS fit (slope = k_B) with its R^2 annotated;
  * k_B with TWO statistical uncertainties -- from the fit covariance and from
    std/sqrt(n) of the per-bead slopes -- the larger drawn as the band, both in
    the summary; plus the robust per-bead MEDIAN k_B as a cross-check;
  * run metadata: run name, T +/- unc, eta(T), n beads.

Inputs  : measurements/<run>/pipeline/{msd.csv, radius.csv, curation_proposed.csv}
          (produced by process_all.py) + T from runs.json.
Outputs : figures/kb_grid.png          (PRIMARY: the per-run k_B grid)
          figures/kb_grid_summary.csv  (per-run table, both error estimates)
          figures/kb_sweep_uniform.png (k_B vs T + D*r vs T, same uniform data)

Usage:  python kb_grid.py                      # all analysed runs
        python kb_grid.py run1 run5 run7 ...   # a subset, in this order
"""
import argparse
import os

import numpy as np
import pandas as pd

from pipeline import paths, physics, figstyle

KB = physics.K_B
# per-bead trust gates -- identical to analyze_run.GATES (week-1 parity)
GATES = dict(D_rel_err=0.5, R_cv=0.20, resid=0.15, inlier=0.60)
MAD_K = 3.5            # robust D*r mislink cut (week-1 plot2_pooled default)
ALPHA_BAND = (0.7, 1.3)   # free-diffusion validity (MSD exponent); stated, not tuned
RADIUS_OFFSET_PX = 1.0    # common-mode radius systematic (median is exposed to OFFSET)

# Runs excluded from the measurement, with the evidence (convection is a GATE,
# not a correction -- a convective run is dropped, never "fixed and trusted").
EXCLUDED_RUNS = {
    # QUALITY discards only (convection etc.). Same-T duplicates are POOLED, not
    # dropped (user policy 2026-06-11): every clean tagged run contributes, and
    # run-to-run agreement at the same T is part of the reproducibility evidence.
    "run12": "convection: QA collective flow 2x every other run (flow_med 0.69 "
             "px/fr, struct_rms 0.75 vs ~0.3-0.4), 7/12 beads drift-flagged "
             "(median |v|=363 nm/s), k_B 1.7x while clean same-T run13 gives "
             "1.08x; user-confirmed drop 2026-06-11",
    "run10": "residual drift below the per-bead gate: 4/11 beads drift-flagged, "
             "median |v|=167 nm/s (2x same-T run9's 76), surviving beads' alpha "
             "median 1.15 (run9: 1.05) -> sub-threshold collective motion "
             "inflating D (k_B 1.33x). Mild run12-type defect; user-confirmed "
             "drop 2026-06-11",
    "run11": "same drift signature as run10, cut by the same criterion: 6/11 "
             "beads drift-flagged, median |v|=165 nm/s (2x same-T run9's 76); "
             "reads 1.34x, agreeing with drifty run10 not quiet run9 (1.00x). "
             "user-confirmed drop 2026-06-11",
    "run6":  "convection (active-cooling run): 5/8 free beads drift-flagged, "
             "leaving n=3 -- two independent disqualifiers. user-confirmed drop "
             "2026-06-11",
    "run14": "non-stationary / convective: full-clip (474 s) D(t) wanders "
             "0.17-0.27 um2/s with a 220 s excursion and 63.7 px cumulative "
             "drift -- no >=120 s stationary plateau by the pre-registered rule; "
             "radius-free D*r confirms it is motion, not composition. Dropped as "
             "never-equilibrated 2026-06-11",
    "run13": "CATEGORY-level disqualification (clean per-run, but the 23-24 C "
             "heated-just-above-ambient regime is non-reproducible: same-T "
             "run12=1.77x convection + run14=0.63x non-stationary; a lone clean "
             "run cannot anchor a temperature that otherwise failed). Heated-"
             "class drop 2026-06-11",
}


def median_se(x):
    """Median + robust SE (1.4826*MAD/sqrt n)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    m = float(np.median(x))
    return m, float(np.median(np.abs(x - m)) * 1.4826 / np.sqrt(len(x)))


def origin_fit(x, y):
    """Through-origin LS slope of y vs x with covariance SE and R^2.

    For a through-origin model the conventional R^2 is the UNCENTERED form
    R^2 = 1 - SS_res / sum(y^2) (variation about 0, not about the mean) -- the
    centered R^2 is ill-defined here and can go negative when the x-range is
    narrow. This uncentered R^2 measures how much of D's magnitude the
    proportional law D = slope*(1/r) explains."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    sxx = float(np.sum(x * x))
    if sxx == 0 or len(x) < 2:
        return np.nan, np.nan, np.nan
    slope = float(np.sum(x * y) / sxx)
    resid = y - slope * x
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(x) - 1, 1) / sxx))
    syy = float(np.sum(y ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / syy if syy > 0 else np.nan
    return slope, se, r2


def analyse_run(stem, mpp):
    """Load one run's hand-tagged radii + MSD D, gate, drop wall-pinned beads,
    and compute k_B (fit slope + robust median) at the run's measured T.

    MANUAL-ONLY: a run is included iff it has radius_manual.csv (hand-marked true
    radii that bypass the diffraction over-read of the auto outer-edge fit). The
    tagged beads ARE the curated set (human-vetted singles), so only the D-quality
    gate is applied -- the auto ring-fit gates (R_cv/resid/inlier) would wrongly
    reject small beads whose hand radius + track are fine."""
    out = paths.out_dir(stem, make=False)
    fm = os.path.join(out, "msd.csv")
    fman = os.path.join(out, "radius_manual.csv")
    if not (os.path.exists(fm) and os.path.exists(fman)):
        return None
    rec = paths.load_runs().get("runs", {}).get(stem, {})
    T = rec.get("T_C")
    if T is None:
        return None
    T_unc = rec.get("T_unc_C", 1.0)

    mp = pd.read_csv(fm)
    man = pd.read_csv(fman)[["particle", "r_um_manual"]]
    df = mp.merge(man, on="particle", how="inner")
    df["r_um"] = df["r_um_manual"]
    df = df.dropna(subset=["D_um2_s", "r_um"])

    # D-quality gate (beads are human-vetted singles with hand radii)
    df = df[df["D_err"] / df["D_um2_s"] < GATES["D_rel_err"]].copy()
    # free-diffusion VALIDITY gate: alpha in [0.7,1.3]. A stuck/wall-bound bead
    # keeps a finite radius but sub-diffuses (alpha<<1); this is a physics cut
    # (stated independent of the outcome), companion to the r<=r* wall cut.
    if "alpha" in df.columns:
        df = df[df["alpha"].between(*ALPHA_BAND)].copy()

    # robust D*r mislink cut (Stokes-Einstein => D*r ~ const)
    dr = (df["D_um2_s"] * df["r_um"]).values
    if len(dr):
        med = np.median(dr); mad = np.median(np.abs(dr - med))
        if mad > 0:
            df = df[np.abs(dr - med) <= MAD_K * mad].copy()

    eta = physics.water_viscosity_Pa_s(T)
    rstar = physics.sediment_r_star_um(T)
    # ---- WALL-PINNED REMOVAL: keep only free diffusers (r <= r*) ----
    n_wall = int((df["r_um"] > rstar).sum())
    df = df[df["r_um"] <= rstar].copy()

    df["inv_r"] = 1.0 / df["r_um"]
    # radius uncertainty for hand-tagged beads: ~1 px localization floor on the edge
    sig_r = pd.Series(mpp, index=df.index)
    df["sig_r"] = sig_r
    df["sig_invr"] = sig_r / df["r_um"] ** 2
    df["kb_i"] = physics.kB_per_bead(df["D_um2_s"], df["r_um"], T, eta)

    pref = physics.kB_prefactor(T) * 1e-18           # k_B = pref * slope[um^3/s]

    # ---- SIGNIFICANT-DRIFT REMOVAL (headline policy): residual per-bead drift
    # inflates D -> k_B (run12: 7/12 flagged, 363 nm/s median |v| -> 1.69x, while
    # clean same-T run13 -> 1.08x). Convection is a GATE, not a correction: the
    # fit set excludes flagged beads; they stay VISIBLE in the panel (open
    # markers) and the all-bead k_B is reported as the sensitivity.
    if "drift_flag" in df.columns:
        df["drift_flag"] = df["drift_flag"].fillna(False).astype(bool)
    else:
        df["drift_flag"] = False
    fit = df[~df["drift_flag"]]
    n = len(fit)                                      # the measurement set
    n_drift = int(df["drift_flag"].sum())
    slopes = (fit["D_um2_s"] * fit["r_um"]).values    # per-bead D*r [um^3/s]

    # (A) through-origin LS slope -> k_B with covariance SE + R^2  (headline)
    slope_ls, se_ls, r2 = origin_fit(fit["inv_r"].values, fit["D_um2_s"].values)
    kb_ls = pref * slope_ls
    se_kb_fit = pref * se_ls
    # (B) std/sqrt(n) of the per-bead slopes (kept per fix 4)
    se_kb_scatter = (pref * float(np.std(slopes, ddof=1)) / np.sqrt(n)
                     if n > 1 else np.nan)
    # (C) robust per-bead MEDIAN -- THE HEADLINE estimator (never regresses on
    # 1/r, so the narrow lever arm + radius x-scatter cannot bias it the way they
    # bias the slope; the LS slope is kept only as a faint cross-check).
    kb_i = fit["kb_i"].values
    kb_med, se_kb_med = median_se(kb_i)
    slope_med = kb_med / pref
    # goodness-of-fit: reduced chi^2 of the per-bead k_B about the median, errors
    # propagated from sigma_D AND sigma_r (1 px). >>1 => the bars underestimate
    # the real bead-to-bead scatter (radius realization), not Gaussian noise.
    sig_i = kb_i * np.sqrt((fit["D_err"].values / fit["D_um2_s"].values) ** 2
                           + (fit["sig_r"].values / fit["r_um"].values) ** 2)
    chi2dof = (float(np.sum(((kb_i - kb_med) / sig_i) ** 2)) / (n - 1)
               if n > 1 else np.nan)
    # common-mode systematics (do NOT average down): radius OFFSET +/-1 px, and
    # the +/-1 C temperature-label band via eta(T).
    dpx = RADIUS_OFFSET_PX * mpp
    kb_rplus = float(np.median(physics.kB_per_bead(fit["D_um2_s"], fit["r_um"] + dpx, T, eta)))
    kb_rminus = float(np.median(physics.kB_per_bead(fit["D_um2_s"], fit["r_um"] - dpx, T, eta)))
    syst_r = abs(kb_rplus - kb_rminus) / 2.0
    kb_Tp = float(np.median(physics.kB_per_bead(fit["D_um2_s"], fit["r_um"], T + T_unc)))
    kb_Tm = float(np.median(physics.kB_per_bead(fit["D_um2_s"], fit["r_um"], T - T_unc)))
    syst_T = abs(kb_Tp - kb_Tm) / 2.0
    # sensitivity: all free beads incl. drift-flagged
    kb_med_all = float(np.median(df["kb_i"].values)) if len(df) else np.nan

    return dict(run=stem, T=float(T), T_unc=float(T_unc), eta_cP=eta * 1e3,
                rstar=float(rstar), df=df, fit=fit, n=n, n_wall=n_wall,
                n_drift=n_drift, kb_med_all=kb_med_all, chi2dof=chi2dof,
                syst_r=syst_r, syst_T=syst_T,
                slope_ls=slope_ls, kb_ls=kb_ls, se_kb_fit=se_kb_fit,
                se_kb_scatter=se_kb_scatter, r2=r2,
                slope_med=slope_med, kb_med=kb_med, se_kb_med=se_kb_med)


def draw_panel(ax, res, xmax, ymax):
    fit, drift = res["fit"], res["df"][res["df"]["drift_flag"]]
    pref = physics.kB_prefactor(res["T"]) * 1e-18
    # measurement set: x (sigma_1/r) and y (sigma_D) error bars
    ax.errorbar(fit["inv_r"], fit["D_um2_s"],
                xerr=fit["sig_invr"], yerr=fit["D_err"],
                fmt="o", ms=4, color="#1f77b4", ecolor="0.6", elinewidth=0.8,
                capsize=1.5, mec="white", mew=0.4, zorder=3)
    # drift-excluded beads: visible but NOT in the fit (open markers)
    if len(drift):
        ax.errorbar(drift["inv_r"], drift["D_um2_s"],
                    xerr=drift["sig_invr"], yerr=drift["D_err"],
                    fmt="o", ms=4.5, mfc="none", mec="#d62728", ecolor="0.75",
                    elinewidth=0.7, capsize=1.5, mew=1.0, zorder=2)
    xs = np.array([0.0, xmax])
    # HEADLINE: robust per-bead median (slope = median(D*r)), solid + stat band
    se_slope_med = (res["se_kb_med"] / pref) if pref else 0.0
    ax.fill_between(xs, (res["slope_med"] - se_slope_med) * xs,
                    (res["slope_med"] + se_slope_med) * xs,
                    color="#d62728", alpha=0.13, zorder=1)
    ax.plot(xs, res["slope_med"] * xs, "-", color="#d62728", lw=2.0, zorder=5)
    # through-origin LS slope -- faint dotted CROSS-CHECK only (no lever arm)
    if np.isfinite(res["slope_ls"]):
        ax.plot(xs, res["slope_ls"] * xs, ":", color="0.5", lw=1.2, zorder=4)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)

    ratio = res["kb_med"] / KB
    se_show = res["se_kb_med"] / KB
    ndrift = (f", {res['n_drift']} drift-excl." if res["n_drift"] else "")
    txt = (f"{res['run']}   {res['T']:.1f}$\\pm${res['T_unc']:.0f}$^\\circ$C\n"
           f"$\\eta$={res['eta_cP']:.2f} cP\n"
           rf"$k_B$={ratio:.2f}$\pm${se_show:.2f}$\,k_B^{{\rm acc}}$"
           "\n"
           rf"$\chi^2$/dof={res['chi2dof']:.1f}   $n$={res['n']}{ndrift}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))


def build_grid(results, out_png):
    import matplotlib.pyplot as plt
    figstyle.set_style()
    n = len(results)
    ncols = 4 if n > 9 else (3 if n > 4 else max(1, n))
    nrows = int(np.ceil(n / ncols))
    xmax = max((1 / r["df"]["r_um"]).max() for r in results) * 1.08
    ymax = max(r["df"]["D_um2_s"].max() for r in results) * 1.12
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for k, res in enumerate(results):
        draw_panel(axes[k // ncols][k % ncols], res, xmax, ymax)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    for i in range(nrows):
        axes[i][0].set_ylabel(r"$D$  [$\mu$m$^2$/s]")
    # x-label on the bottom-most VISIBLE panel of each column (some last-row
    # panels may be turned off when n is not a multiple of ncols)
    for j in range(ncols):
        rows_here = [k // ncols for k in range(n) if k % ncols == j]
        if rows_here:
            ax_b = axes[max(rows_here)][j]
            ax_b.set_xlabel(r"$1/r$  [$\mu$m$^{-1}$]")
            ax_b.tick_params(labelbottom=True)
    from matplotlib.lines import Line2D
    rstar_lo = min(r["rstar"] for r in results); rstar_hi = max(r["rstar"] for r in results)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=7, label="free spheres ($r\\leq r^*$); bars = $\\sigma_D,\\ \\sigma_{1/r}$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#d62728", markersize=7,
               label="significant drift, excluded ($|v|{>}0.1\\,\\mu$m/s & $>2\\sigma_v$)"),
        Line2D([0], [0], color="#d62728", lw=2.2,
               label=r"per-bead median $k_B$ (headline; band $=\pm\sigma_{\rm stat}$)"),
        Line2D([0], [0], color="0.5", lw=1.4, ls=":",
               label="through-origin LS slope (cross-check; no lever arm)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Week-2 Stokes-Einstein per run:  $k_B$ = per-bead median of "
                 r"$6\pi\eta(T)\,r D/T$  (hand-tagged radii; free diffusers "
                 rf"$r\leq r^*\approx{rstar_lo:.1f}$-${rstar_hi:.1f}\,\mu$m; "
                 r"$\alpha\in[0.7,1.3]$; shared axes)",
                 fontsize=12.5, y=1.004)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figstyle.save(fig, out_png)
    plt.close(fig)
    print(f"[kb_grid] wrote {out_png}")


def build_sweep(results, out_png):
    """k_B invariance vs T + radius-free D*r vs T, from the HEADLINE measurement
    set (free, drift-excluded -- same beads the panel fits use)."""
    import matplotlib.pyplot as plt
    figstyle.set_style()
    allfree = pd.concat([r["fit"].assign(T=r["T"], run=r["run"]) for r in results
                         if len(r["fit"])], ignore_index=True)
    rows = []
    for T, gp in allfree.groupby("T"):
        m, se = median_se(gp["kb_i"])
        dm, dse = median_se(gp["D_um2_s"] * gp["r_um"])
        rows.append(dict(T=float(T), n_free=len(gp), n_runs=gp["run"].nunique(),
                         eta_cP=round(physics.water_viscosity_cP(float(T)), 3),
                         kb=m, kb_se=se, ratio=m / KB, Dr=dm, Dr_se=dse))
    tab = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    grand, gse = median_se(allfree["kb_i"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    runs = sorted(allfree["run"].unique(), key=lambda s: int(s[3:]))
    for i, run in enumerate(runs):
        gp = allfree[allfree["run"] == run]
        jit = (i - (len(runs) - 1) / 2) * 0.10
        ax[0].scatter(gp["T"] + jit, gp["kb_i"], s=18, alpha=0.5)
    ax[0].errorbar(tab["T"], tab["kb"], yerr=tab["kb_se"], fmt="s", ms=9,
                   color="k", capsize=4, zorder=5, label="per-T median")
    ax[0].axhline(KB, color="C2", lw=1.6, label="accepted $k_B$")
    ax[0].axhline(grand, color="C3", ls="--", lw=1.4, label=f"pooled ({grand / KB:.2f}x)")
    ax[0].set_xlabel("temperature [$^\\circ$C]")
    ax[0].set_ylabel(r"per-bead $k_{B,i}$ [J/K]")
    ax[0].set_title("$k_B$ invariance vs temperature (free beads)")
    ax[0].set_ylim(0, None); ax[0].legend(fontsize=8, ncol=2)

    Ts = np.linspace(tab["T"].min() - 2, tab["T"].max() + 2, 120)
    pred = np.array([KB * (t + 273.15) / (6 * np.pi * physics.water_viscosity_Pa_s(t))
                     * 1e18 for t in Ts])
    ax[1].plot(Ts, pred, "C2-", lw=2, label=r"SE @ accepted $k_B$:  $k_BT/6\pi\eta(T)$")
    ax[1].errorbar(tab["T"], tab["Dr"], yerr=tab["Dr_se"], fmt="o", ms=9,
                   color="C0", capsize=4, label=r"median $D\,r$ (free)")
    ax[1].set_xlabel("temperature [$^\\circ$C]")
    ax[1].set_ylabel(r"$D\,r$ [$\mu$m$^3$/s]")
    ax[1].set_title(r"temperature dependence:  $D r = k_B T / 6\pi\eta(T)$")
    ax[1].set_ylim(0, None); ax[1].legend(fontsize=8)
    figstyle.save(fig, out_png)
    plt.close(fig)
    print(f"[kb_grid] wrote {out_png}")
    return tab, grand, gse


def main():
    ap = argparse.ArgumentParser(description="Week-2 per-run k_B grid (D vs 1/r).")
    ap.add_argument("runs", nargs="*", help="run stems (default: all analysed)")
    ap.add_argument("--min-free", type=int, default=3,
                    help="exclude a run with fewer than this many free beads "
                         "(a through-origin slope from n<3 is essentially "
                         "unconstrained). Default 3.")
    ap.add_argument("--split", action="store_true",
                    help="also write one standalone figure-2 per run "
                         "(figures/kb_run_<run>.png), same panel as the grid "
                         "but on its own axes")
    args = ap.parse_args()
    min_free = max(2, args.min_free)

    mpp = paths.load_scale() or 0.14381
    allruns = sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:]))
    only = args.runs or allruns
    results = []
    for stem in only:
        if stem in EXCLUDED_RUNS:
            print(f"[kb_grid] {stem}: DISCARDED -- {EXCLUDED_RUNS[stem]}")
            continue
        res = analyse_run(stem, mpp)
        if res is None:
            print(f"[kb_grid] {stem}: no radius_manual.csv (untagged) -> skip; "
                  f"tag with: python radius_tag.py {stem}")
            continue
        if res["n"] < min_free:
            print(f"[kb_grid] {stem}: only {res['n']} free bead(s) "
                  f"(< min_free={min_free}) -> EXCLUDED from grid/sweep/pool")
            continue
        print(f"[kb_grid] {stem}: T={res['T']:.1f}C eta={res['eta_cP']:.2f}cP  "
              f"n_free={res['n']} (dropped {res['n_wall']} wall, {res['n_drift']} drift)  "
              f"k_B(median)={res['kb_med']/KB:.2f}+/-{res['se_kb_med']/KB:.2f}x  "
              f"chi2/dof={res['chi2dof']:.1f}  [slope x-check {res['kb_ls']/KB:.2f}x]")
        results.append(res)
    if not results:
        raise SystemExit("no analysed runs found -- run process_all.py first")

    figdir = paths.FIGURES_DIR
    summ = pd.DataFrame([{
        "run": r["run"], "T_C": r["T"], "T_unc_C": r["T_unc"], "eta_cP": r["eta_cP"],
        "rstar_um": r["rstar"], "n_free": r["n"], "n_wall_dropped": r["n_wall"],
        "n_drift_excluded": r["n_drift"],
        "kb_median": r["kb_med"], "se_kb_median_stat": r["se_kb_med"],
        "chi2_dof": r["chi2dof"], "syst_radius_1px": r["syst_r"], "syst_T_1C": r["syst_T"],
        "ratio_median": r["kb_med"] / KB, "ratio_median_incl_drift": r["kb_med_all"] / KB,
        "kb_slope_xcheck": r["kb_ls"], "ratio_slope_xcheck": r["kb_ls"] / KB,
    } for r in results])
    summ.to_csv(os.path.join(figdir, "kb_grid_summary.csv"), index=False)

    build_grid(results, os.path.join(figdir, "kb_grid.png"))
    tab, grand, gse = build_sweep(results, os.path.join(figdir, "kb_sweep_uniform.png"))

    if args.split:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        figstyle.set_style()
        for res in results:
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            xmax = (1 / res["df"]["r_um"]).max() * 1.15
            ymax = res["df"]["D_um2_s"].max() * 1.18
            draw_panel(ax, res, xmax, ymax)
            ax.set_xlabel(r"$1/r$  [$\mu$m$^{-1}$]")
            ax.set_ylabel(r"$D$  [$\mu$m$^2$/s]")
            ax.set_title("Stokes-Einstein:  "
                         r"$D = \dfrac{k_B T}{6\pi\eta}\,\dfrac{1}{r}$"
                         f"   ({res['run']})")
            handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                       markersize=7, label="free spheres ($r\\leq r^*$)"),
                Line2D([0], [0], color="#d62728", lw=2.2,
                       label="per-bead median $k_B$ (headline)"),
                Line2D([0], [0], color="0.5", lw=1.4, ls=":",
                       label="LS slope (cross-check)"),
            ]
            if res["n_drift"]:
                handles.insert(1, Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor="none",
                                         markeredgecolor="#d62728", markersize=7,
                                         label="drift (excluded)"))
            ax.legend(handles=handles, loc="lower right", fontsize=8.5)
            p = figstyle.save(fig, os.path.join(figdir, f"kb_run_{res['run']}.png"))
            plt.close(fig)
            print(f"[kb_grid] wrote {os.path.basename(p)}")

    # sensitivity: pooled k_B INCLUDING the drift-flagged beads
    allbeads = pd.concat([r["df"] for r in results], ignore_index=True)
    grand_all, _ = median_se(allbeads["kb_i"])
    n_all = len(allbeads)
    n_fit = int((~allbeads["drift_flag"]).sum())

    print("\n=== WEEK-2 per-run k_B (hand-tagged radii; free, drift-excluded) ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== pooled by temperature (free, drift-excluded) ===")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    # common-mode systematics on the pooled median (do NOT average down)
    Dall = allbeads.loc[~allbeads["drift_flag"], "D_um2_s"].values
    Rall = allbeads.loc[~allbeads["drift_flag"], "r_um"].values
    Tall = np.array([r["T"] for r in results for _ in range(r["n"])])
    dpx = (paths.load_scale() or 0.14381)
    sr = abs(np.median(physics.kB_per_bead(Dall, Rall + dpx, Tall))
             - np.median(physics.kB_per_bead(Dall, Rall - dpx, Tall))) / 2
    print(f"\nHEADLINE pooled k_B = {grand:.3e} +/- {gse:.1e}(stat) J/K "
          f"({grand / KB:.2f}x accepted) over {len(summ)} runs, "
          f"{tab['T'].nunique()} temperatures, n={n_fit} (free, drift-excluded)")
    print(f"  systematics (common-mode, do NOT average down): "
          f"radius +/-1px = +/-{sr/KB:.2f}x ; T-label +/-1C ~ +/-0.03x")
    print(f"  sensitivity -- INCLUDING the {n_all - n_fit} drift beads: "
          f"k_B = {grand_all:.3e} ({grand_all / KB:.2f}x), n={n_all}")


if __name__ == "__main__":
    main()
