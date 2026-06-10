"""
batch.py  (pipeline)
--------------------
End-to-end, fully-automatic batch over all analyzable runs:
  per run:  track -> auto-curate -> contact sheet -> MSD->D -> radius -> k_B
  then:     pool room-temp runs for the HEADLINE k_B,
            pool all runs, and a TEMPERATURE-SERIES k_B-invariance check.

No human curation: run3 uses your confirmed labels.csv; every other run uses
the validated auto-curation gates (purity-first). run2 is discarded.

Temperatures: room runs = 25 C; run7,run8 = 15 C; run9,run10 = 5 C. These are
the AIMED temperatures -- runs.json has no measured values -- so the cold-run
k_B is PROVISIONAL until real thermocouple readings are supplied. eta(T) is the
water-viscosity fit in physics.py.

Robust: each run is wrapped in try/except so one bad clip can't kill the batch.

Usage:  python -m pipeline.batch
"""

import os
import traceback
import numpy as np
import pandas as pd

from . import (paths, track, curate, contact_sheet, msd, radius, aggregate,
               physics, figstyle)

TEMP_C = {"run3": 25.0, "run4": 25.0, "run5": 25.0, "run6": 25.0,
          "run7": 15.0, "run8": 15.0, "run9": 5.0, "run10": 5.0}
ROOM = ["run3", "run4", "run5", "run6"]
ALL = ["run3", "run4", "run5", "run6", "run7", "run8", "run9", "run10"]
TRACK_RUNS = ["run4", "run5", "run6", "run7", "run8", "run9", "run10"]  # run3 done
DELTA_RHO = 60.0
KB = physics.K_B


def process_run(stem):
    """track (if needed) -> curate -> contact sheet -> msd -> radius -> aggregate."""
    out = paths.out_dir(stem)
    if not os.path.exists(os.path.join(out, "trajectory.csv")):
        track.run(stem)
    curate.run(stem)
    try:
        contact_sheet.render(stem)        # morning spot-check material; non-fatal
    except Exception:                      # noqa: BLE001
        traceback.print_exc()
    msd.run(stem)
    radius.run(stem)
    try:
        aggregate.run(stem, temp_C=TEMP_C[stem], delta_rho=DELTA_RHO)
    except SystemExit as e:
        print(f"[batch] {stem}: single-run aggregate skipped ({e})")


def run_beads(stem):
    """Per-bead trustworthy (D, r, kb_i) for one run's curated singles, with the
    run's temperature folded into eta + kb_i. None if unavailable."""
    out = paths.out_dir(stem)
    mp, rp, tp = (os.path.join(out, f) for f in
                  ("msd.csv", "radius.csv", "trajectory.csv"))
    if not (os.path.exists(mp) and os.path.exists(rp)):
        return None
    df = pd.read_csv(mp).merge(pd.read_csv(rp), on="particle", how="inner")
    df = df.dropna(subset=["D_um2_s", "r_um"])
    kept = curate.kept_pids(out)
    if kept is not None:
        df = df[df["particle"].isin(kept)]
    df = df[(df["D_err"] / df["D_um2_s"] < 0.5) & (df["R_cv"] < 0.20)
            & (df["resid_med"] < 0.15) & (df["inlier_med"] > 0.60)].copy()
    if os.path.exists(tp) and len(df):
        keep, _ = aggregate.dedup_fragments(set(df["particle"].astype(int)),
                                            pd.read_csv(tp))
        df = df[df["particle"].isin(keep)].copy()
    if not len(df):
        return None
    T = TEMP_C[stem]
    eta = physics.water_viscosity_Pa_s(T)
    df["run"] = stem
    df["T"] = T
    df["kb_i"] = physics.kB_per_bead(df["D_um2_s"], df["r_um"], T, eta)
    df["rstar"] = physics.sediment_r_star_um(T, DELTA_RHO)
    df["free"] = df["r_um"] <= df["rstar"]
    return df


def _median_kb(kb):
    kb = np.asarray(kb, float)
    med = float(np.median(kb))
    se = float(np.median(np.abs(kb - med)) * 1.4826 / np.sqrt(len(kb)))
    return med, se


def pool(runs, label, fname):
    import matplotlib.pyplot as plt
    frames = [run_beads(r) for r in runs]
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        print(f"[batch] pool '{label}': no beads")
        return None
    a = pd.concat(frames, ignore_index=True)
    free = a[a["free"]]
    if len(free) < 3:
        print(f"[batch] pool '{label}': only {len(free)} free beads -- skipping")
        return None
    med, se = _median_kb(free["kb_i"])
    med_all, _ = _median_kb(a["kb_i"])

    figstyle.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for r, g in a.groupby("run"):
        ax[0].scatter(1 / g["r_um"], g["D_um2_s"], s=26, alpha=0.65,
                      label=f"{r} (n={len(g)})")
    xs = np.linspace(0, (1 / free["r_um"]).max() * 1.05, 50)
    pref = physics.kB_prefactor(25.0) * 1e-18
    ax[0].plot(xs, med / pref * xs, "k-", lw=2, label=f"median ({med/KB:.2f}x)")
    ax[0].set_xlabel(r"$1/r$ [$\mu$m$^{-1}$]"); ax[0].set_ylabel(r"$D$ [$\mu$m$^2$/s]")
    ax[0].set_xlim(0, None); ax[0].set_ylim(0, None); ax[0].legend(fontsize=7)
    ax[0].set_title(f"{label}: D vs 1/r (n_free={len(free)})")
    sc = ax[1].scatter(a["r_um"], a["kb_i"], s=28, c=a["T"], cmap="coolwarm",
                       edgecolor="white", lw=0.4)
    ax[1].axhline(KB, color="k", lw=1.4, label="accepted $k_B$")
    ax[1].axhline(med, color="C2", ls="--", lw=1.6, label=f"free median ({med/KB:.2f}x)")
    ax[1].axhspan(med - se, med + se, color="C2", alpha=0.12)
    ax[1].set_xlabel(r"$r$ [$\mu$m]"); ax[1].set_ylabel(r"per-bead $k_{B,i}$ [J/K]")
    ax[1].legend(fontsize=8); ax[1].set_title("per-bead $k_B$ vs size (colour = T)")
    fig.colorbar(sc, ax=ax[1], fraction=0.046, label="T [C]")
    p = figstyle.save(fig, os.path.join(paths.FIGURES_DIR, fname))
    plt.close(fig)
    print(f"[batch] '{label}': n_free={len(free)}/{len(a)}  "
          f"k_B={med:.3e} ({med/KB:.3f}x) +/- {se:.2e} ({se/med*100:.0f}%); "
          f"all-singles {med_all/KB:.3f}x -> {p}")
    return dict(label=label, n_free=int(len(free)), n=int(len(a)), kb=med, se=se,
                ratio=med / KB, ratio_all=med_all / KB,
                per_run={r: int((a["run"] == r).sum()) for r in a["run"].unique()})


def temperature_series(runs, fname="temperature_series.png"):
    import matplotlib.pyplot as plt
    frames = [run_beads(r) for r in runs]
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    a = pd.concat(frames, ignore_index=True)
    free = a[a["free"]]
    rows = []
    for T, g in free.groupby("T"):
        if len(g) >= 3:
            med, se = _median_kb(g["kb_i"])
            rows.append((float(T), med, se, int(len(g))))
    grand, gse = _median_kb(free["kb_i"]) if len(free) >= 3 else (np.nan, np.nan)

    figstyle.set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5))
    if rows:
        Ts, meds, ses, ns = zip(*rows)
        ax.errorbar(Ts, meds, yerr=ses, fmt="o", ms=9, capsize=5, color="C0",
                    label="free median per T")
        for T, m, s, n in rows:
            ax.annotate(f"n={n}", (T, m), textcoords="offset points",
                        xytext=(8, 6), fontsize=8)
    ax.axhline(KB, color="k", lw=1.5, label=f"accepted $k_B$={KB:.3e}")
    if np.isfinite(grand):
        ax.axhline(grand, color="C3", ls="--", lw=1.4,
                   label=f"grand pooled ({grand/KB:.2f}x)")
    ax.set_xlabel("temperature [C]  (15/5 = AIMED, not measured)")
    ax.set_ylabel(r"$k_B$ [J/K]")
    ax.set_title("k_B invariance across temperature"); ax.legend(fontsize=8)
    ax.set_ylim(0, None)
    p = figstyle.save(fig, os.path.join(paths.FIGURES_DIR, fname))
    plt.close(fig)
    print(f"[batch] temperature series -> {p}; grand pooled k_B={grand:.3e} "
          f"({grand/KB:.3f}x)" + (f"; per-T={[(t, round(m/KB,2)) for t,m,_,_ in rows]}" if rows else ""))
    return rows, grand, gse


def main():
    print("=" * 74)
    print(f"[batch] runs: track {TRACK_RUNS} + reuse run3  (run2 discarded)")
    print(f"[batch] temps (AIMED): {TEMP_C}")
    print("=" * 74)
    for stem in TRACK_RUNS:
        print(f"\n--------------------- [batch] {stem} ---------------------")
        try:
            process_run(stem)
        except Exception:                  # noqa: BLE001
            traceback.print_exc()
            print(f"[batch] {stem} FAILED -- continuing")
    print("\n--------------------- [batch] refresh run3 ---------------------")
    try:
        curate.run("run3"); msd.run("run3"); radius.run("run3")
        aggregate.run("run3", temp_C=25.0, delta_rho=DELTA_RHO)
    except Exception:                      # noqa: BLE001
        traceback.print_exc()

    print("\n" + "=" * 74 + "\n[batch] POOLING + TEMPERATURE SERIES\n" + "=" * 74)
    room = pool(ROOM, "room-temp pooled (25C)", "pooled_kb_room.png")
    allp = pool(ALL, "all-runs pooled", "pooled_kb_all.png")
    ts = temperature_series(ALL)

    with open(os.path.join(paths.FIGURES_DIR, "batch_summary.txt"), "w") as f:
        f.write("BATCH SUMMARY  (run2 discarded; T for run7-10 is AIMED 15/5 C, "
                "not measured -> cold-run k_B provisional)\n")
        f.write(f"accepted k_B = {KB:.4e} J/K\n\n")
        for tag, res in (("ROOM-TEMP (run3,4,5,6)", room), ("ALL RUNS", allp)):
            if res:
                f.write(f"{tag}: k_B = {res['kb']:.4e} ({res['ratio']:.3f}x accepted), "
                        f"n_free={res['n_free']}/{res['n']}, +/-{res['se']:.2e}; "
                        f"all-singles contrast {res['ratio_all']:.3f}x\n")
                f.write(f"    per-run beads: {res['per_run']}\n")
        if ts and ts[0]:
            f.write("\nTEMPERATURE SERIES (free median per T):\n")
            for T, m, s, n in ts[0]:
                f.write(f"    {T:.0f} C: {m:.4e} ({m/KB:.3f}x)  n={n}\n")
            f.write(f"  grand pooled: {ts[1]:.4e} ({ts[1]/KB:.3f}x)\n")
    print(f"\n[batch] wrote batch_summary.txt + pooled/temperature figures -> "
          f"{paths.FIGURES_DIR}")
    print("[batch] DONE")


if __name__ == "__main__":
    main()
