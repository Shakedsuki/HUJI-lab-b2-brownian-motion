"""
aggregate.py  (pipeline)
------------------------
Confirmed singles' (D, r) -> Boltzmann constant via Stokes-Einstein, with an
honest error budget. The accuracy bottleneck is the radius; the precision/bias
trap is the estimator.

ESTIMATOR (headline): per-bead k_B,i = 6 pi eta r_i D_i / T, pooled as the
MEDIAN over the free-diffusion set. This is equal-weight + robust. We also print:
  - Theil-Sen slope of D vs 1/r (robust slope, independent check)
  - through-origin least-squares slope (the NAIVE estimator) -- shown precisely
    because it weights large-1/r (small, biased, noisy) beads heavily and reads
    high; reproducing it documents the ~19% trap rather than hiding it.

FREE-DIFFUSION CUT: Stokes-Einstein assumes unbounded 3-D diffusion. A bead
pinned at a coverslip sub-diffuses (wall drag), biasing D -- and k_B -- low, size-
dependently. A bead is a free diffuser while its gravitational length exceeds its
radius: r < r* = [k_B T / ((4/3) pi |drho| g)]^(1/4). For polyethylene (buoyant,
rho<water) the bead rises to the TOP wall; r* depends only on |drho| (no k_B
input -> not circular). Headline over r <= r*; sensitivity to the cut is in the
budget.

ERROR BUDGET: statistical (MAD/sqrt n) + temperature band + size-cut + the
radius diffraction-bias band (shift r by -delta_px, since the outer edge
overestimates the true radius).
"""

import os
import numpy as np
import pandas as pd

from . import physics


def _median_kb(D, r, T_C, eta=None):
    kb = physics.kB_per_bead(D, r, T_C, eta)
    med = float(np.median(kb))
    se = float(np.median(np.abs(kb - med)) * 1.4826 / np.sqrt(len(kb)))
    return med, se, kb


def _slope_origin(D, r):
    """Through-origin LS slope of D vs (1/r): Sxy/Sxx [um^3/s]."""
    x = 1.0 / r
    return float(np.sum(x * D) / np.sum(x * x))


def dedup_fragments(pids, traj, max_gap=25, max_jump=15.0):
    """Collapse linker-break FRAGMENTS of one physical bead (one track ends, a
    near-identical one begins a few frames later at the same de-drifted spot) to
    a single bead, keeping the longest fragment -- so a bead isn't counted N
    times in the k_B statistics. Returns (kept set, n_dropped)."""
    pids = list(pids)
    info = {}
    for p in pids:
        g = traj[traj["particle"] == p].sort_values("frame")
        info[p] = (int(g["frame"].iloc[0]), int(g["frame"].iloc[-1]),
                   g[["x", "y"]].iloc[0].to_numpy(),
                   g[["x", "y"]].iloc[-1].to_numpy(), len(g))
    parent = {p: p for p in pids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in pids:
        _, f1a, _, p1a, _ = info[a]
        for b in pids:
            if b == a:
                continue
            f0b, _, p0b, _, _ = info[b]
            if 0 <= f0b - f1a <= max_gap and np.hypot(*(p0b - p1a)) <= max_jump:
                parent[find(a)] = find(b)
    groups = {}
    for p in pids:
        groups.setdefault(find(p), []).append(p)
    keep = {max(g, key=lambda p: info[p][4]) for g in groups.values()}
    return keep, len(pids) - len(keep)


def run(stem, temp_C=25.0, eta_cP=None, delta_rho=60.0, r_lo=0.0, r_hi=None,
        t_band=2.0, bias_px=1.0, videos_dir=None):
    from . import paths, figstyle, curate
    import matplotlib.pyplot as plt

    out = paths.out_dir(stem)
    msd = pd.read_csv(os.path.join(out, "msd.csv"))
    rad = pd.read_csv(os.path.join(out, "radius.csv"))
    df = msd.merge(rad, on="particle", how="inner",
                   suffixes=("_msd", "_rad")).dropna(subset=["D_um2_s", "r_um"])
    kept = curate.kept_pids(out)
    if kept is not None:
        df = df[df["particle"].isin(kept)]
    if len(df) < 3:
        raise SystemExit(f"[agg] only {len(df)} beads with D+r -- need >=3")

    # reliability gates: even a human-kept single needs a TRUSTWORTHY D + radius
    # for the SE fit (a defocused single has a bad radius; a short track a noisy
    # D). These are separate from the morphological 'single' label.
    n0 = len(df)
    df = df[(df["D_err"] / df["D_um2_s"] < 0.5)
            & (df["R_cv"] < 0.20) & (df["resid_med"] < 0.15)
            & (df["inlier_med"] > 0.60)].copy()
    n_relgate = n0 - len(df)
    # de-duplicate linker-break fragments of the same physical bead
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    keep_set, n_dup = dedup_fragments(set(df["particle"].astype(int)), traj)
    df = df[df["particle"].isin(keep_set)]
    print(f"[agg] reliability gate dropped {n_relgate}, fragment dedup dropped "
          f"{n_dup} -> {len(df)} beads with trustworthy D+r")
    if len(df) < 3:
        raise SystemExit(f"[agg] only {len(df)} beads after reliability gates")

    eta = eta_cP * 1e-3 if eta_cP is not None else physics.water_viscosity_Pa_s(temp_C)
    mpp = paths.load_scale() or 0.14381
    rstar = physics.sediment_r_star_um(temp_C, delta_rho)
    r_hi = rstar if r_hi is None else r_hi
    free = df[(df["r_um"] >= r_lo) & (df["r_um"] <= r_hi)].copy()
    print(f"[agg] {stem}: {len(df)} singles; free-diffusion cut r<= {r_hi:.2f}um "
          f"(sediment r*={rstar:.2f}um, |drho|={delta_rho:.0f}); {len(free)} free")
    if len(free) < 3:
        raise SystemExit(f"[agg] only {len(free)} free beads -- widen --r-hi")

    D, r = free["D_um2_s"].to_numpy(), free["r_um"].to_numpy()
    med, se, kb_i = _median_kb(D, r, temp_C, eta)
    med_all = _median_kb(df["D_um2_s"].to_numpy(), df["r_um"].to_numpy(),
                         temp_C, eta)[0]                  # no free cut, for contrast
    pref = physics.kB_prefactor(temp_C, eta) * 1e-18      # k_B = pref * slope[um^3/s]
    kb_ls = pref * _slope_origin(D, r)
    # Theil-Sen is only meaningful with enough points; degenerate (even negative)
    # at tiny n, so guard it.
    kb_ts = np.nan
    if len(D) >= 5:
        try:
            from scipy.stats import theilslopes
            kb_ts = pref * theilslopes(D, 1.0 / r)[0]
        except Exception:                                 # noqa: BLE001
            kb_ts = np.nan

    # ---- error budget --------------------------------------------------------
    m_loT = _median_kb(D, r, temp_C - t_band, None)[0]
    m_hiT = _median_kb(D, r, temp_C + t_band, None)[0]
    syst_T = abs(m_loT - m_hiT) / 2.0
    cut_meds = []
    for c in [rstar - 0.2, rstar, rstar + 0.3, rstar + 0.5]:
        sub = df[(df["r_um"] >= r_lo) & (df["r_um"] <= c)]
        if len(sub) >= 5:
            cut_meds.append(_median_kb(sub["D_um2_s"].to_numpy(),
                                       sub["r_um"].to_numpy(), temp_C, eta)[0])
    syst_cut = (max(cut_meds) - min(cut_meds)) / 2.0 if len(cut_meds) > 1 else 0.0
    # radius diffraction bias: true r is SMALLER by ~bias_px -> k_B lower
    dbias = bias_px * mpp
    med_biascorr = _median_kb(D, np.clip(r - dbias, 1e-3, None), temp_C, eta)[0]
    syst_bias = abs(med - med_biascorr)
    tot = float(np.sqrt(se ** 2 + syst_T ** 2 + syst_cut ** 2 + syst_bias ** 2))

    kB = physics.K_B
    print("=" * 72)
    print(f"  POOLED FREE SINGLES  n={len(free)}/{len(df)}   T={temp_C}C  "
          f"eta={eta*1e3:.3f}cP")
    ts_str = (f"{kb_ts:.3e}  ratio={kb_ts/kB:.3f}" if np.isfinite(kb_ts)
              else "n/a (need >=5 free beads)")
    print(f"  k_B (per-bead MEDIAN)   = {med:.3e}  ratio={med/kB:.3f}   <- HEADLINE")
    print(f"  k_B (all singles,no cut)= {med_all:.3e}  ratio={med_all/kB:.3f}   (contrast: wall bias)")
    print(f"  k_B (Theil-Sen slope)   = {ts_str}   (robust check)")
    print(f"  k_B (through-0 LS slope)= {kb_ls:.3e}  ratio={kb_ls/kB:.3f}   (naive trap)")
    print(f"  k_B (median, bias-corr -{bias_px:.0f}px) = {med_biascorr:.3e}  "
          f"ratio={med_biascorr/kB:.3f}")
    print(f"  error budget:")
    print(f"    stat (MAD/sqrt n)   = +/- {se:.2e}  ({se/med*100:.1f}%)")
    print(f"    T-band (+/-{t_band:.0f}C)      = +/- {syst_T:.2e}  ({syst_T/med*100:.1f}%)")
    print(f"    size-cut (r* +/-)   = +/- {syst_cut:.2e}  ({syst_cut/med*100:.1f}%)")
    print(f"    radius bias (-{bias_px:.0f}px)  = +/- {syst_bias:.2e}  ({syst_bias/med*100:.1f}%)")
    print(f"  => k_B = ({med/1e-23:.2f} +/- {tot/1e-23:.2f})e-23 J/K   "
          f"[{med/kB:.2f} +/- {tot/kB:.2f} x accepted]")
    print("=" * 72)

    df["kb_i"] = physics.kB_per_bead(df["D_um2_s"], df["r_um"], temp_C, eta)
    df.to_csv(os.path.join(out, "kb_per_bead.csv"), index=False)
    with open(os.path.join(out, "kb_summary.txt"), "w") as f:
        f.write(f"{stem}: k_B median = {med:.4e} J/K ({med/kB:.3f} x accepted)\n"
                f"n_free={len(free)}/{len(df)}, T={temp_C}C, eta={eta*1e3:.3f}cP, "
                f"r*={rstar:.2f}um\n"
                f"total unc = {tot:.3e} ({tot/kB:.3f} x); stat={se:.2e} T={syst_T:.2e} "
                f"cut={syst_cut:.2e} bias={syst_bias:.2e}\n"
                f"Theil-Sen={kb_ts:.4e} ({kb_ts/kB:.3f}x), "
                f"naive-LS={kb_ls:.4e} ({kb_ls/kB:.3f}x)\n")

    # ---- figure --------------------------------------------------------------
    figstyle.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    excl = df[df["r_um"] > r_hi]
    ax[0].scatter(1 / excl["r_um"], excl["D_um2_s"], s=28, facecolor="none",
                  edgecolor="0.7", label=f"wall-excluded r>r* (n={len(excl)})")
    ax[0].scatter(1 / free["r_um"], free["D_um2_s"], s=42, color="C0",
                  edgecolor="white", lw=0.5, label=f"free singles (n={len(free)})")
    xs = np.linspace(0, (1 / free["r_um"]).max() * 1.05, 50)
    ax[0].plot(xs, kb_ls / pref * xs, "r--", lw=1.4,
               label=f"naive LS ({kb_ls/kB:.2f}x)")
    ax[0].plot(xs, med / pref * xs, "k-", lw=2.0,
               label=f"median ({med/kB:.2f}x)")
    ax[0].set_xlabel(r"$1/r$ [$\mu$m$^{-1}$]"); ax[0].set_ylabel(r"$D$ [$\mu$m$^2$/s]")
    ax[0].set_xlim(0, None); ax[0].set_ylim(0, None)
    ax[0].set_title(f"{stem}: Stokes-Einstein, free singles"); ax[0].legend(fontsize=8)

    ax[1].scatter(df["r_um"], df["kb_i"], s=40, color="C0", edgecolor="white", lw=0.5)
    ax[1].axhline(kB, color="k", lw=1.4, label="accepted $k_B$")
    ax[1].axhline(med, color="C3", ls="--", lw=1.6, label=f"median ({med/kB:.2f}x)")
    ax[1].axhspan(med - tot, med + tot, color="C3", alpha=0.12)
    ax[1].axvspan(r_lo, r_hi, color="C2", alpha=0.08)
    ax[1].axvline(rstar, color="C2", ls=":", lw=1.5, label=f"$r^*$={rstar:.2f}$\\mu$m")
    ax[1].set_xlabel(r"$r$ [$\mu$m]")
    ax[1].set_ylabel(r"per-bead $k_{B,i}=6\pi\eta r_i D_i/T$ [J/K]")
    ax[1].set_title("per-bead $k_B$ vs size + free window"); ax[1].legend(fontsize=8)
    p = figstyle.save(fig, os.path.join(out, "plot2_kb.png"))
    plt.close(fig)
    print(f"[agg] wrote kb_per_bead.csv, kb_summary.txt, plot2_kb.png -> {out}")
    return dict(k_B=med, ratio=med / kB, total_unc=tot, n_free=len(free))


if __name__ == "__main__":   # python -m pipeline.aggregate run3
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate D+r -> k_B.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--temp-C", type=float, default=25.0)
    ap.add_argument("--eta-cP", type=float, default=None)
    ap.add_argument("--delta-rho", type=float, default=60.0,
                    help="|bead-water| density diff [kg/m^3]; polyethylene ~37-87")
    ap.add_argument("--r-lo", type=float, default=0.0)
    ap.add_argument("--r-hi", type=float, default=None)
    ap.add_argument("--bias-px", type=float, default=1.0)
    args = ap.parse_args()
    run(args.run, temp_C=args.temp_C, eta_cP=args.eta_cP, delta_rho=args.delta_rho,
        r_lo=args.r_lo, r_hi=args.r_hi, bias_px=args.bias_px)
