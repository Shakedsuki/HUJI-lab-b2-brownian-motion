"""
aggregate.py  (week1-system-calibration)
----------------------------------------
Pool the curated single spheres across room-temperature runs and report one
combined k_B, restricted to FREE DIFFUSERS, with an honest error budget.

CURATION (objective, reproducible, never uses D or k_B):
  round + clean circle fit (circ_resid_frac, inlier_frac), well tracked
  (n_frames), focus-stable (r_px_frame_cv), bounded MSD intercept.

FREE-DIFFUSION SIZE CUT (the key physics):
  Stokes-Einstein D = k_B T / (6 pi eta r) assumes UNBOUNDED 3-D diffusion. Dense
  beads sediment toward a coverslip and then diffuse slower (wall / Faxen drag)
  -> D too low -> k_B,i biased low for large beads. A bead is a free diffuser
  only while its gravitational length exceeds its radius:
        l_g(r) = k_B T / (Delta_rho g (4/3) pi r^3) > r
        => free while  r < r* = [ k_B T / ((4/3) pi Delta_rho g) ]^(1/4)
  r* depends only on bead size + density (NO k_B input -> not circular). For
  polystyrene (Delta_rho ~ 50 kg/m^3) r* ~ 1.2 um, and the measured k_B-vs-r
  trend turns over right there. We headline the median k_B over r <= r* (the free
  set), show the full trend + the cut, and quote the sensitivity to the cut.

ESTIMATOR: per-bead k_B,i = 6 pi eta r_i D_i / T; pooled headline = MEDIAN over
the free set (robust; the trend is ~flat there so median ~ slope-fit).

Usage
-----
    python scripts/aggregate.py --runs run2 run3 run4
    python scripts/aggregate.py --delta-rho 60        # polyethylene
    python scripts/aggregate.py --r-free-hi 1.5       # override the cut
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


def r_star_um(temp_C, delta_rho):
    """Radius [um] where gravitational length equals radius (free-diffusion boundary)."""
    return physics.sedimentation_r_star_um(temp_C, delta_rho)


def load_curated(cdir, run, gates):
    rp, mp = os.path.join(cdir, "radius.csv"), os.path.join(cdir, "msd.csv")
    if not (os.path.exists(rp) and os.path.exists(mp)):
        return None
    rad, msd = pd.read_csv(rp), pd.read_csv(mp)
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
    return df


def median_kb(df, temp_C, eta_cP=None):
    if len(df) == 0:
        return np.nan, np.nan
    pref = pref_of(temp_C, eta_cP)
    kb = pref * df["D_um2_s"].values * df["r"].values
    med = float(np.median(kb))
    mad = float(np.median(np.abs(kb - med))) * 1.4826
    return med, mad / np.sqrt(len(kb))


def slope_kb(df, temp_C, eta_cP=None):
    return slope_kb_se(df, temp_C, eta_cP)[0]


def slope_kb_se(df, temp_C, eta_cP=None):
    """Through-origin LS k_B and its statistical SE for D vs 1/r."""
    pref = pref_of(temp_C, eta_cP)
    x, y = df["invr"].values, df["D_um2_s"].values
    slope = float(np.sum(x * y) / np.sum(x * x))
    resid = y - slope * x
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(x) - 1, 1) / np.sum(x * x)))
    return pref * slope, pref * se


def main():
    ap = argparse.ArgumentParser(description="Aggregate k_B across room-temp runs (free diffusers).")
    ap.add_argument("--runs", nargs="+", default=["run2", "run3", "run4"])
    ap.add_argument("--temp-C", type=float, default=25.0)
    ap.add_argument("--eta-cP", type=float, default=None)
    ap.add_argument("--t-lo", type=float, default=21.0)
    ap.add_argument("--t-hi", type=float, default=25.0)
    ap.add_argument("--delta-rho", type=float, default=50.0,
                    help="bead-fluid density diff [kg/m^3] (polystyrene ~50, polyethylene ~60)")
    ap.add_argument("--r-free-lo", type=float, default=0.0,
                    help="lower radius bound [um] (curation already drops tiny beads)")
    ap.add_argument("--r-free-hi", type=float, default=None,
                    help="upper radius bound [um]; default = sedimentation r*")
    args = ap.parse_args()

    frames = []
    for run in args.runs:
        d = load_curated(_paths.clip_dir(run), run, GATES)
        if d is None or len(d) == 0:
            print(f"[agg] {run}: no curated beads (missing radius/msd or all gated out) -- skipping")
            continue
        frames.append(d)
    if not frames:
        sys.exit("[agg] nothing to aggregate -- run msd_fit + measure_radius on the runs first")
    alld = pd.concat(frames, ignore_index=True)

    rstar = r_star_um(args.temp_C, args.delta_rho)
    r_hi = args.r_free_hi if args.r_free_hi is not None else rstar
    free = alld[(alld["r"] >= args.r_free_lo) & (alld["r"] <= r_hi)]
    if len(free) < 3:
        sys.exit(f"[agg] only {len(free)} free-diffuser beads (r<= {r_hi:.2f}um) -- widen --r-free-hi")

    med_all, _ = median_kb(alld, args.temp_C, args.eta_cP)
    med, se = median_kb(free, args.temp_C, args.eta_cP)
    slope = slope_kb(free, args.temp_C, args.eta_cP)
    m_hiT, _ = median_kb(free, args.t_hi, args.eta_cP)   # higher T -> lower k_B
    m_loT, _ = median_kb(free, args.t_lo, args.eta_cP)
    syst_T = abs(m_loT - m_hiT) / 2.0
    # size-cut sensitivity: median over alternative upper cuts around r*
    cut_meds = []
    for c in [rstar - 0.2, rstar, rstar + 0.3, rstar + 0.5]:
        sub = alld[(alld["r"] >= args.r_free_lo) & (alld["r"] <= c)]
        if len(sub) >= 5:
            cut_meds.append(median_kb(sub, args.temp_C, args.eta_cP)[0])
    syst_cut = (max(cut_meds) - min(cut_meds)) / 2.0 if len(cut_meds) > 1 else 0.0
    tot = np.sqrt(se ** 2 + syst_T ** 2 + syst_cut ** 2)

    print(f"[agg] sedimentation r* = {rstar:.2f} um (Delta_rho={args.delta_rho:.0f} kg/m^3); free cut r <= {r_hi:.2f} um")
    for run, g in alld.groupby("run"):
        gf = g[g["r"] <= r_hi]
        mr = median_kb(gf, args.temp_C, args.eta_cP)[0]
        print(f"      {run}: {len(g):3d} curated, {len(gf):3d} free   free median k_B={mr:.3e} ({mr/physics.K_B:.2f}x)")
    print("=" * 70)
    print(f"POOLED FREE DIFFUSERS: n={len(free)}/{len(alld)}   (T={args.temp_C} C)")
    print(f"  k_B (per-bead median) = {med:.3e}  ratio={med/physics.K_B:.3f}   <- HEADLINE")
    print(f"  k_B (slope-fit)       = {slope:.3e}  ratio={slope/physics.K_B:.3f}   (agreement check)")
    print(f"  all-bead median (no cut, for contrast) = {med_all:.3e}  ratio={med_all/physics.K_B:.3f}")
    print(f"  error budget:")
    print(f"    stat (median SE)      = +/- {se:.2e}  ({se/med*100:.1f}%)")
    print(f"    T-band ({args.t_lo:.0f}-{args.t_hi:.0f} C)      = +/- {syst_T:.2e}  ({syst_T/med*100:.1f}%)")
    print(f"    size-cut (r* +/- )    = +/- {syst_cut:.2e}  ({syst_cut/med*100:.1f}%)")
    print(f"    => k_B = ({med/1e-23:.2f} +/- {tot/1e-23:.2f}) x10^-23 J/K   "
          f"[{med/physics.K_B:.2f} +/- {tot/physics.K_B:.2f} x accepted]")
    print("=" * 70)

    # ---- figure: D vs 1/r (free fit) + per-bead k_B vs r (the cut, shown) ----
    figure_style.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    pref = pref_of(args.temp_C, args.eta_cP)
    colors = {r: f"C{i}" for i, r in enumerate(args.runs)}

    excl = alld[alld["r"] > r_hi]
    ax[0].scatter(excl["invr"], excl["D_um2_s"], s=26, facecolor="none", edgecolor="0.7",
                  zorder=2, label=f"wall-excluded, $r>r^*$ (n={len(excl)})")
    for run, g in free.groupby("run"):
        ax[0].scatter(g["invr"], g["D_um2_s"], s=42, color=colors.get(run, "C7"),
                      edgecolor="white", lw=0.5, zorder=3, label=f"{run} free (n={len(g)})")
    xs = np.linspace(0, free["invr"].max() * 1.05, 100)
    ax[0].plot(xs, slope * xs, "k-", lw=2.2, zorder=4,
               label=f"free fit: $k_B$={slope:.2e} ({slope/physics.K_B:.2f}$\\times$)")
    ax[0].set_xlabel(r"$1/r$  [$\mu$m$^{-1}$]"); ax[0].set_ylabel(r"$D$  [$\mu$m$^2$/s]")
    ax[0].set_title(f"Pooled Stokes–Einstein, free diffusers  (N={len(free)})")
    ax[0].set_xlim(0, None); ax[0].set_ylim(0, None); ax[0].legend(fontsize=8)

    for run, g in alld.groupby("run"):
        ax[1].scatter(g["r"], pref * g["D_um2_s"] * g["r"], s=40, color=colors.get(run, "C7"),
                      edgecolor="white", lw=0.5, zorder=3, label=f"{run} (n={len(g)})")
    ax[1].axhline(physics.K_B, color="k", lw=1.5, label="accepted $k_B$")
    ax[1].axhline(med, color="C3", ls="--", lw=1.6, label=f"free median ({med/physics.K_B:.2f}$\\times$)")
    ax[1].axhspan(med - se, med + se, color="C3", alpha=0.12)
    ax[1].axvspan(args.r_free_lo, r_hi, color="C2", alpha=0.08)
    ax[1].axvline(rstar, color="C2", ls=":", lw=1.5, label=f"sediment. $r^*$={rstar:.2f} $\\mu$m")
    ax[1].set_xlabel(r"$r$  [$\mu$m]")
    ax[1].set_ylabel(r"per-bead $k_{B,i}=6\pi\eta r_i D_i/T$  [J/K]")
    ax[1].set_title("Per-bead $k_B$ vs size: free-diffusion window")
    ax[1].legend(fontsize=8, loc="upper right")

    path = figure_style.savefig(f"aggregate_{'_'.join(args.runs)}.png", fig=fig)
    plt.close(fig)
    print(f"[agg] wrote {path}")


if __name__ == "__main__":
    main()
