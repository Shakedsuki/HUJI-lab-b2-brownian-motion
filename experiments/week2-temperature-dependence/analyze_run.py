"""
analyze_run.py <run> [<run> ...]
--------------------------------
Single-run analysis (one video at a time): auto-curate -> MSD/D -> radius ->
per-bead k_B at the run's MEASURED temperature (T_C = starting_temp, +/-1 C).

Reuses the validated week-pipeline stages; temperature comes from runs.json
(never tuned to k_B). Writes measurements/<run>/pipeline/kb_per_bead.csv and a
D-vs-1/r figure with the Stokes-Einstein prediction at this T.

  k_B,i = 6 pi eta(T) r_i D_i / T      (per bead)
  free-diffusion cut: r_i <= r*(T, delta_rho)  (sedimentation length; not circular)
"""
import os
import sys

import numpy as np
import pandas as pd

from pipeline import paths, curate, msd, radius, physics

# radius ground-truth bracket from the CPMS-0.96 stock label (1-10 um diameter
# -> radius 0.5-5 um). Delta_rho now comes from the MEASURED 0.96 g/cc density at
# T (physics.delta_rho_kg_m3), replacing week1's inherited 60.
R_SPEC_UM = (0.5, 5.0)
# per-bead trust gates (identical to week1 batch.run_beads)
GATES = dict(D_rel_err=0.5, R_cv=0.20, resid=0.15, inlier=0.60)


def analyze(stem):
    runs = paths.load_runs().get("runs", {})
    rec = runs.get(stem, {})
    T = rec.get("T_C")
    T_unc = rec.get("T_unc_C", 1.0)
    if T is None:
        raise SystemExit(f"{stem}: no T_C in runs.json")
    out = paths.out_dir(stem)
    if not os.path.exists(os.path.join(out, "trajectory.csv")):
        raise SystemExit(f"{stem}: no trajectory.csv -- track first")

    print(f"\n=== analyze {stem}: T = {T} +/- {T_unc} C ===", flush=True)
    if os.path.exists(os.path.join(out, "curation.csv")):
        print("[analyze] reusing existing curation.csv (labels.csv if present)",
              flush=True)
    else:
        curate.run(stem)
    msd.run(stem)
    radius.run(stem)

    mp = pd.read_csv(os.path.join(out, "msd.csv"))
    rp = pd.read_csv(os.path.join(out, "radius.csv"))
    df = mp.merge(rp, on="particle", how="inner").dropna(subset=["D_um2_s", "r_um"])
    man_path = os.path.join(out, "radius_manual.csv")
    man_used = os.path.exists(man_path)
    if man_used:
        # hand-tagged radii (radius_tag.py) bypass the diffraction over-read.
        # The tagged set IS the curated set -> use it directly, override r_um.
        man = pd.read_csv(man_path)
        df = df.merge(man[["particle", "r_um_manual"]], on="particle", how="inner")
        df["r_um"] = df["r_um_manual"]
        df["r_px_med"] = df["r_um"] / (paths.load_scale() or 0.14381)
        print(f"[analyze] using {len(df)} MANUAL radii (radius_manual.csv); "
              f"auto outer-ring radii overridden", flush=True)
        # Re-apply curation's IDENTITY/contamination flags to the seeded set.
        # Human seeding happens on ONE (sharpest) frame, so it cannot see a
        # DYNAMIC mislink or a resolved rigid doublet that only shows over the
        # track. We re-impose curate.py's contamination verdicts (two-cores,
        # mislink, rigid-doublet) -- but NOT the auto-ring-fit QUALITY gates
        # (R_cv/resid/inlier), which reject good small beads on a poor AUTO fit
        # that the hand radius legitimately bypasses.
        cur_path = os.path.join(out, "curation.csv")
        if os.path.exists(cur_path):
            cur = pd.read_csv(cur_path)
            contam = set(cur[cur["reason"].fillna("").str.contains(
                "two-cores|mislink|rigid", regex=True)]["particle"].astype(int))
            n0 = len(df)
            df = df[~df["particle"].isin(contam)]
            print(f"[analyze] curation re-applied to seeded set: dropped "
                  f"{n0 - len(df)} contaminated (doublet/mislink) bead(s)",
                  flush=True)
    else:
        kept = curate.kept_pids(out)
        if kept is not None:
            df = df[df["particle"].isin(kept)]
    # quality gates. With HAND radii the bead is human-vetted (clean single + true
    # size), so the AUTO ring-fit gates (R_cv/resid/inlier) are inappropriate: they
    # reject small beads whose auto fit is poor but whose hand radius + track are
    # fine. Keep only the D (MSD-fit) quality gate when manual radii are used.
    gate = df["D_err"] / df["D_um2_s"] < GATES["D_rel_err"]
    if not man_used:
        gate &= ((df["R_cv"] < GATES["R_cv"]) & (df["resid_med"] < GATES["resid"])
                 & (df["inlier_med"] > GATES["inlier"]))
    df = df[gate].copy()

    eta = physics.water_viscosity_Pa_s(T)
    drho = physics.delta_rho_kg_m3(T)
    df["T"] = T
    df["eta_cP"] = eta * 1e3
    df["kb_i"] = physics.kB_per_bead(df["D_um2_s"], df["r_um"], T, eta)
    rstar = physics.sediment_r_star_um(T)        # Delta_rho from measured 0.96 g/cc
    df["r_star_um"] = rstar
    df["free"] = df["r_um"] <= rstar
    # DIAGNOSTIC ONLY (never a filter): a free single sphere cannot diffuse
    # faster than unbounded Stokes-Einstein, so D/D_SE > 1 is unphysical. But
    # D_SE uses the ACCEPTED k_B, so cutting on it would tune us to the answer
    # and (since noise scatters ~half of even-perfect beads above 1) bias low.
    # We FLAG egregious violators (no plausible k_B makes D/D_SE > ~1.5 physical)
    # for inspection; we do not drop them.
    df["D_SE_um2_s"] = physics.stokes_einstein_D(T, df["r_um"] * 1e-6, eta) * 1e12
    df["D_over_SE"] = df["D_um2_s"] / df["D_SE_um2_s"]
    n_egreg = int((df["D_over_SE"] > 1.5).sum())
    if n_egreg:
        print(f"[analyze] FLAG (not cut): {n_egreg} bead(s) with D/D_SE > 1.5 "
              f"-- inspect for radius over-tag or mislink", flush=True)
    df.to_csv(os.path.join(out, "kb_per_bead.csv"), index=False)

    free = df[df["free"]]
    KB = physics.K_B
    # HEADLINE = week1 method: median of per-bead k_B,i = 6 pi eta r D / T over ALL
    # clean singles (equivalently median(D_i*r_i) * 6 pi eta / T), NO free cut --
    # exactly what week1 reported (0.96x). Including the large wall-hindered beads
    # is deliberate: their low D*r partially offsets the radius over-read, the same
    # cancellation week1 relied on. The free-only median is a secondary diagnostic.
    kb_all = float(np.median(df["kb_i"])) if len(df) >= 3 else np.nan
    kb_free = float(np.median(free["kb_i"])) if len(free) >= 3 else np.nan
    n_over = int((df["r_um"] > R_SPEC_UM[1]).sum())
    print(f"[analyze] {stem}: {len(df)} clean singles ({len(free)} free <r*={rstar:.1f}um, "
          f"dRho={drho:.0f}); eta={eta*1e3:.3f}cP; "
          f"r_um {df['r_um'].min():.2f}-{df['r_um'].max():.2f} ({n_over} over spec-max); "
          f"k_B(all singles, WEEK1) = {kb_all:.3e} J/K ({kb_all/KB:.2f}x)  <- HEADLINE"
          + (f"; free-only = {kb_free:.3e} ({kb_free/KB:.2f}x)"
             if np.isfinite(kb_free) else "; free-only n<3"),
          flush=True)
    _plot(stem, df, free, T, rstar, kb_all, out)
    return dict(run=stem, T=T, n=len(df), n_free=len(free), eta_cP=eta * 1e3,
                kb=kb_all, ratio=kb_all / KB if np.isfinite(kb_all) else np.nan,
                kb_free=kb_free)


def _plot(stem, df, free, T, rstar, kb_med, out):
    import matplotlib.pyplot as plt
    from pipeline import figstyle
    figstyle.set_style()
    KB = physics.K_B
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # D vs 1/r with SE prediction at accepted k_B
    ax[0].scatter(1 / df["r_um"], df["D_um2_s"], s=26, c=df["free"].map(
        {True: "C0", False: "0.6"}), label="beads (grey=wall-pinned)")
    xs = np.linspace(0, (1 / df["r_um"]).max() * 1.05, 50)
    pref = physics.kB_prefactor(T) * 1e-18         # D[um2/s] = KB/(pref) * (1/r[um])
    ax[0].plot(xs, KB / pref * xs, "k--", lw=1.6,
               label=r"SE @ accepted $k_B$")
    if np.isfinite(kb_med):
        ax[0].plot(xs, kb_med / pref * xs, "C2-", lw=2,
                   label=rf"SE @ median all singles ({kb_med/KB:.2f}x)")
    ax[0].set_xlabel(r"$1/r$ [$\mu$m$^{-1}$]"); ax[0].set_ylabel(r"$D$ [$\mu$m$^2$/s]")
    ax[0].set_xlim(0, None); ax[0].set_ylim(0, None); ax[0].legend(fontsize=8)
    ax[0].set_title(f"{stem}: D vs 1/r  (T={T} C)")
    # per-bead k_B vs size
    ax[1].scatter(df["r_um"], df["kb_i"], s=28,
                  c=df["free"].map({True: "C0", False: "0.6"}))
    ax[1].axhline(KB, color="k", lw=1.4, label="accepted $k_B$")
    if np.isfinite(kb_med):
        ax[1].axhline(kb_med, color="C2", ls="--", lw=1.6, label="median (all singles)")
    ax[1].axvline(rstar, color="C3", ls=":", lw=1.2, label=f"r*={rstar:.1f}um")
    ax[1].set_xlabel(r"$r$ [$\mu$m]"); ax[1].set_ylabel(r"per-bead $k_{B,i}$ [J/K]")
    ax[1].set_ylim(0, None); ax[1].legend(fontsize=8)
    ax[1].set_title("per-bead $k_B$ vs size")
    p = figstyle.save(fig, os.path.join(out, "kb_run.png"))
    plt.close(fig)
    print(f"[analyze] wrote kb_per_bead.csv + {os.path.basename(p)} -> {out}",
          flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["run7"]):
        analyze(s)
