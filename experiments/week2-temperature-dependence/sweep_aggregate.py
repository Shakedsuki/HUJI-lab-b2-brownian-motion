"""
sweep_aggregate.py
------------------
Pool per-run kb_per_bead.csv across temperatures into the week-2 result views:

  (A) per-bead k_B vs T  -- the INVARIANCE test. k_B is a constant of nature, so
      if D, r, T and eta(T) are all right the points lie FLAT at accepted k_B.
      A tilt with T would betray a systematic in eta(T) or T; the absolute level
      reflects the residual radius/D offset (a ~constant offset cancels here).

  (B) median D*r vs T  -- the radius-free Stokes-Einstein group: D*r = k_B T /
      6 pi eta(T), independent of how we sized each bead. Overlaid with the
      accepted-k_B prediction curve, this shows directly whether the diffusion
      follows the SE temperature dependence.

Writes figures/sweep_kb.png + figures/sweep_summary.csv. Run after >=1 run is
analyzed (meaningful from 2+ temperatures).

Usage:  python sweep_aggregate.py
"""
import glob
import os

import numpy as np
import pandas as pd

from pipeline import paths, physics

KB = physics.K_B


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(paths.MEAS_DIR, "*", "pipeline",
                                           "kb_per_bead.csv"))):
        d = pd.read_csv(p)
        d["run"] = p.split(os.sep)[-3]
        rows.append(d)
    if not rows:
        raise SystemExit("no kb_per_bead.csv found -- analyze at least one run first")
    return pd.concat(rows, ignore_index=True)


def median_se(x):
    """Median + robust standard error (1.4826*MAD / sqrt n)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    m = float(np.median(x))
    return m, float(np.median(np.abs(x - m)) * 1.4826 / np.sqrt(len(x)))


def main():
    a = load()
    free = a[a["free"]].copy()
    if not len(free):
        raise SystemExit("no free beads across runs yet")

    rows = []
    for T, g in free.groupby("T"):
        m, se = median_se(g["kb_i"])
        dm, dse = median_se(g["D_um2_s"] * g["r_um"])      # D*r [um^3/s]
        rows.append(dict(T=float(T), n_free=len(g), n_runs=g["run"].nunique(),
                         eta_cP=round(physics.water_viscosity_cP(float(T)), 3),
                         kb=m, kb_se=se, ratio=m / KB, Dr=dm, Dr_se=dse))
    tab = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    grand, gse = median_se(free["kb_i"])

    print("=== WEEK-2 TEMPERATURE SWEEP (free beads) ===")
    print(tab.to_string(index=False,
                        float_format=lambda v: f"{v:.4g}"))
    print(f"\ngrand pooled k_B = {grand:.3e} +/- {gse:.1e} J/K "
          f"({grand / KB:.2f}x accepted); n_free={len(free)} over "
          f"{free['run'].nunique()} run(s), {tab['T'].nunique()} temperature(s)")

    import matplotlib.pyplot as plt
    from pipeline import figstyle
    figstyle.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    runs = sorted(free["run"].unique(), key=lambda s: int(s[3:]))
    for i, run in enumerate(runs):
        g = free[free["run"] == run]
        jit = (i - (len(runs) - 1) / 2) * 0.12               # separate same-T runs
        ax[0].scatter(g["T"] + jit, g["kb_i"], s=20, alpha=0.5, label=run)
    if len(tab):
        ax[0].errorbar(tab["T"], tab["kb"], yerr=tab["kb_se"], fmt="s", ms=9,
                       color="k", capsize=4, zorder=5, label="per-T median")
    ax[0].axhline(KB, color="C2", lw=1.6, label="accepted $k_B$")
    ax[0].axhline(grand, color="C3", ls="--", lw=1.4,
                  label=f"grand pooled ({grand / KB:.2f}x)")
    ax[0].set_xlabel("temperature [C]")
    ax[0].set_ylabel(r"per-bead $k_{B,i}$ [J/K]")
    ax[0].set_title("k_B invariance vs temperature")
    ax[0].set_ylim(0, None)
    ax[0].legend(fontsize=7, ncol=2)

    Ts = np.linspace(free["T"].min() - 2, free["T"].max() + 2, 120)
    pred = np.array([KB * (T + 273.15)
                     / (6 * np.pi * physics.water_viscosity_Pa_s(T)) * 1e18
                     for T in Ts])                            # um^3/s
    ax[1].plot(Ts, pred, "C2-", lw=2,
               label=r"SE @ accepted $k_B$:  $k_B T / 6\pi\eta(T)$")
    ax[1].errorbar(tab["T"], tab["Dr"], yerr=tab["Dr_se"], fmt="o", ms=9,
                   color="C0", capsize=4, label=r"median $D\,r$ (free)")
    ax[1].set_xlabel("temperature [C]")
    ax[1].set_ylabel(r"$D\,r$ [$\mu$m$^3$/s]")
    ax[1].set_title(r"temperature dependence:  $D r = k_B T / 6\pi\eta(T)$")
    ax[1].set_ylim(0, None)
    ax[1].legend(fontsize=8)

    p = figstyle.save(fig, os.path.join(paths.FIGURES_DIR, "sweep_kb.png"))
    plt.close(fig)
    tab.to_csv(os.path.join(paths.FIGURES_DIR, "sweep_summary.csv"), index=False)
    print(f"wrote sweep_kb.png + sweep_summary.csv -> {paths.FIGURES_DIR}")


if __name__ == "__main__":
    main()
