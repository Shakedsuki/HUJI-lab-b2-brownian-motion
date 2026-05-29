"""
aggregate.py  (week1-system-calibration)
----------------------------------------
Pool the CURATED single spheres across several room-temperature runs and report
one combined k_B with an honest error budget.

CURATION IS OBJECTIVE AND REPRODUCIBLE (no hand-labelling, no k_B-based cuts):
a bead is a "single sphere" if it is ROUND with a clean circle fit
(circ_resid_frac, inlier_frac), well tracked (n_frames), focus-stable
(r_px_frame_cv) and has a bounded MSD intercept. None of these uses D or the
implied k_B, so there is no circularity. labels.csv (if a run has one) is loaded
only as a cross-check and printed, never used to select.

ESTIMATOR: each single sphere gives k_B,i = 6 pi eta r_i D_i / T; the pooled
robust headline is the MEDIAN over all curated beads (immune to the small-bead
edge-bias / large-bead wall-bias tilt and to outliers).

ERROR BUDGET (printed and drawn):
  stat   : MAD-based SE of the pooled median
  T-band : spread of the median over T in [--t-lo, --t-hi]  (eta(T)/T; dominant)
  curat. : |median - through-origin slope-fit|  (radius/curation systematic)

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/aggregate.py                       # run2 run3 run4
    python scripts/aggregate.py --runs run2 run3 run4 run5
    python scripts/aggregate.py --temp-C 25 --t-lo 21 --t-hi 25
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths
import figure_style
import physics


GATES = dict(min_frames=150, max_intercept=0.25, max_resid=0.10,
             min_inlier=0.70, max_rcv=0.15)


def pref_of(temp_C, eta_cP=None):
    eta = (eta_cP if eta_cP is not None else physics.water_viscosity_cP(temp_C)) * 1e-3
    return 6 * np.pi * eta * 1e-18 / (temp_C + 273.15)


def load_curated(cdir, run, gates):
    """Return curated single-sphere rows for one run (objective gates)."""
    rp, mp = os.path.join(cdir, "radius.csv"), os.path.join(cdir, "msd.csv")
    if not (os.path.exists(rp) and os.path.exists(mp)):
        return None
    rad = pd.read_csv(rp)
    msd = pd.read_csv(mp)
    rcols = [c for c in ["particle", "r_um", "circ_resid_frac", "inlier_frac", "r_px_frame_cv"] if c in rad]
    mcols = [c for c in ["particle", "D_um2_s", "D_err", "n_frames", "intercept_um2"] if c in msd]
    df = rad[rcols].merge(msd[mcols], on="particle", how="inner")
    df = df.rename(columns={"r_um": "r"}).dropna(subset=["r", "D_um2_s"])
    keep = ((df.get("n_frames", 1e9) >= gates["min_frames"])
            & (df.get("intercept_um2", 0).abs() < gates["max_intercept"])
            & (df.get("circ_resid_frac", 0) < gates["max_resid"])
            & (df.get("inlier_frac", 1) > gates["min_inlier"])
            & (df.get("r_px_frame_cv", 0) < gates["max_rcv"]))
    df = df[keep].copy()
    df["run"] = run
    df["invr"] = 1.0 / df["r"]
    rel_D = (df["D_err"] / df["D_um2_s"]).abs() if "D_err" in df else 0.0
    rel_r = df["r_px_frame_cv"].fillna(0.0) if "r_px_frame_cv" in df else 0.0
    df["rel_err"] = np.sqrt(rel_D ** 2 + rel_r ** 2)
    # cross-check only: attach manual label if the run has one
    lp = os.path.join(cdir, "labels.csv")
    if os.path.exists(lp):
        df = df.merge(pd.read_csv(lp)[["particle", "type"]], on="particle", how="left")
    return df


def median_kb(df, temp_C, eta_cP=None):
    pref = pref_of(temp_C, eta_cP)
    kb = pref * df["D_um2_s"].values * df["r"].values
    med = float(np.median(kb))
    mad = float(np.median(np.abs(kb - med))) * 1.4826
    return med, mad / np.sqrt(len(kb))


def slope_kb(df, temp_C, eta_cP=None):
    pref = pref_of(temp_C, eta_cP)
    x, y = df["invr"].values, df["D_um2_s"].values
    return pref * float(np.sum(x * y) / np.sum(x * x))


def main():
    ap = argparse.ArgumentParser(description="Aggregate k_B across room-temp runs.")
    ap.add_argument("--runs", nargs="+", default=["run2", "run3", "run4"])
    ap.add_argument("--temp-C", type=float, default=25.0)
    ap.add_argument("--eta-cP", type=float, default=None)
    ap.add_argument("--t-lo", type=float, default=21.0, help="low end of T systematic band")
    ap.add_argument("--t-hi", type=float, default=25.0, help="high end of T systematic band")
    args = ap.parse_args()

    frames = []
    for run in args.runs:
        cdir = _paths.clip_dir(run)
        d = load_curated(cdir, run, GATES)
        if d is None or len(d) == 0:
            print(f"[agg] {run}: no curated beads (missing radius/msd or all gated out) -- skipping")
            continue
        m, se = median_kb(d, args.temp_C, args.eta_cP)
        xcheck = ""
        if "type" in d:
            ns = d["type"].isin(["perfect", "singlet"]).sum()
            xcheck = f"  (manual-label cross-check: {ns}/{len(d)} also perfect/singlet)"
        print(f"[agg] {run}: {len(d):3d} curated singles   median k_B={m:.3e} ({m/physics.K_B:.2f}x){xcheck}")
        frames.append(d)

    if not frames:
        sys.exit("[agg] nothing to aggregate -- run msd_fit + measure_radius on the runs first")
    alld = pd.concat(frames, ignore_index=True)

    med, se = median_kb(alld, args.temp_C, args.eta_cP)
    slope = slope_kb(alld, args.temp_C, args.eta_cP)
    m_lo, _ = median_kb(alld, args.t_hi, args.eta_cP)     # higher T -> lower k_B
    m_hi, _ = median_kb(alld, args.t_lo, args.eta_cP)     # lower  T -> higher k_B
    syst_T = (m_hi - m_lo) / 2.0
    syst_curation = abs(med - slope)

    print("\n" + "=" * 70)
    print(f"POOLED: {len(alld)} single spheres across {len(frames)} runs   (T={args.temp_C} C)")
    print(f"  k_B (per-bead median) = {med:.3e}  ratio={med/physics.K_B:.3f}")
    print(f"  through-origin slope  = {slope:.3e}  ratio={slope/physics.K_B:.3f}")
    print(f"  error budget:")
    print(f"    stat   (median SE)            = +/- {se:.2e}  ({se/med*100:.1f}%)")
    print(f"    T-band ({args.t_lo:.0f}-{args.t_hi:.0f} C)            = +/- {syst_T:.2e}  ({syst_T/med*100:.1f}%)")
    print(f"    curation (median vs slope)    = +/- {syst_curation:.2e}  ({syst_curation/med*100:.1f}%)")
    tot = np.sqrt(se ** 2 + syst_T ** 2 + syst_curation ** 2)
    print(f"    => k_B = ({med/1e-23:.2f} +/- {tot/1e-23:.2f}) x10^-23 J/K   "
          f"[{med/physics.K_B:.2f} +/- {tot/physics.K_B:.2f} x accepted]")
    print("=" * 70)

    # ---- figure: D vs 1/r (left) + per-bead k_B vs r (right) ----
    figure_style.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    pref = pref_of(args.temp_C, args.eta_cP)
    colors = {r: f"C{i}" for i, r in enumerate(args.runs)}
    for run, g in alld.groupby("run"):
        ax[0].scatter(g["invr"], g["D_um2_s"], s=42, color=colors.get(run, "C7"),
                      alpha=0.8, edgecolor="white", lw=0.5, label=f"{run} (n={len(g)})")
        ax[1].scatter(g["r"], pref * g["D_um2_s"] * g["r"], s=42, color=colors.get(run, "C7"),
                      alpha=0.8, edgecolor="white", lw=0.5, label=f"{run} (n={len(g)})")
    xs = np.linspace(1e-3, alld["invr"].max() * 1.05, 200)
    ax[0].plot(xs, (med / pref) * xs, "k-", lw=2.2,
               label=f"pooled median $k_B$={med:.2e} ({med/physics.K_B:.2f}$\\times$)")
    ax[0].plot(xs, slope * xs, "k--", lw=1.3, alpha=0.7,
               label=f"slope-fit ({slope/physics.K_B:.2f}$\\times$)")
    ax[0].set_xlabel(r"$1/r$  [$\mu$m$^{-1}$]"); ax[0].set_ylabel(r"$D$  [$\mu$m$^2$/s]")
    ax[0].set_title(f"Pooled Stokes–Einstein  (N={len(alld)}, T={args.temp_C:.0f} °C)")
    ax[0].set_xlim(0, None); ax[0].set_ylim(0, None); ax[0].legend(fontsize=8)

    ax[1].axhline(physics.K_B, color="k", lw=1.5, label="accepted $k_B$")
    ax[1].axhline(med, color="C3", ls="--", lw=1.6, label=f"pooled median ({med/physics.K_B:.2f}$\\times$)")
    ax[1].axhspan(med - se, med + se, color="C3", alpha=0.12)
    ax[1].set_xlabel(r"$r$  [$\mu$m]")
    ax[1].set_ylabel(r"per-bead $k_{B,i}=6\pi\eta r_i D_i/T$  [J/K]")
    ax[1].set_title("Per-bead $k_B$ vs size  (small→edge bias ↑, large→wall bias ↓; median robust)")
    ax[1].legend(fontsize=8)

    path = figure_style.savefig(f"aggregate_{'_'.join(args.runs)}.png", fig=fig)
    plt.close(fig)
    print(f"[agg] wrote {path}")


if __name__ == "__main__":
    main()
