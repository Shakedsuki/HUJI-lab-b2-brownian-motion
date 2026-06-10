"""
plot1_perbead.py  <run> [<run> ...]
-----------------------------------
Per-bead MSD diagnostic grid (Plot 1, bead-resolved) for the FREE diffusers of a
run -- wall-pinned beads (r > r*(T)) are removed, matching the measurement set.

Each cell is one bead:
  * main panel: time-averaged MSD(tau) with the short-lag linear fit
    MSD = 4 D tau + c; D +/- sigma_D (fit covariance) and the fit R^2 annotated.
    The fit line is SOLID if R^2 >= --r2-thresh, DASHED (flagged) otherwise.
  * inset: the per-bead drift check -- <Dx(tau)> and <Dy(tau)> vs tau with their
    linear fits; the slopes are the residual drift velocities v_dx, v_dy [um/s]
    (on the de-drifted coordinates). A bead with |v_drift| > --drift-thresh gets
    a red, flagged title.

Reuses pipeline.msd (same MSD, D, R^2 and drift estimators as the CSV stage), so
the figure and msd.csv/drift_perbead.csv agree by construction.

Writes measurements/<run>/pipeline/plot1_perbead.png.

Usage:  python plot1_perbead.py run7
        python plot1_perbead.py run3 run6 run7 run9 run12 run13 run15
        python plot1_perbead.py run7 --max-beads 24 --drift-thresh 0.1 --r2-thresh 0.95
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

from pipeline import paths, physics, figstyle
from pipeline import msd as msdmod


def free_curated_pids(out, T, min_len):
    """Hand-tagged singles that are FREE diffusers (r_manual <= r*(T)). Prefers
    radius_manual.csv (the measurement set); falls back to auto radius + proposal."""
    rstar = physics.sediment_r_star_um(T)
    fman = os.path.join(out, "radius_manual.csv")
    if os.path.exists(fman):
        man = pd.read_csv(fman).rename(columns={"r_um_manual": "r_um"})
        man = man[man["r_um"] <= rstar]
        return man.set_index("particle")["r_um"].to_dict(), rstar
    rad = pd.read_csv(os.path.join(out, "radius.csv"))
    prop = os.path.join(out, "curation_proposed.csv")
    if os.path.exists(prop):
        kept = set(pd.read_csv(prop)["particle"].astype(int))
        rad = rad[rad["particle"].isin(kept)]
    rad = rad[rad["r_um"] <= rstar]
    return rad.set_index("particle")["r_um"].to_dict(), rstar


def run(stem, max_beads=24, min_len=60, max_lag=100, fit_lag=30,
        drift_thresh=0.1, r2_thresh=0.95):
    out = paths.out_dir(stem, make=False)
    if not os.path.exists(os.path.join(out, "msd.csv")):
        print(f"[plot1_perbead] {stem}: no msd.csv -> skip")
        return None
    rec = paths.load_runs().get("runs", {}).get(stem, {})
    T = rec.get("T_C"); T_unc = rec.get("T_unc_C", 1.0)
    mpp = paths.load_scale() or 0.14381
    fps = paths.fps_of(paths.video_for_run(stem)) or 9.30
    dt = 1.0 / fps

    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    counts = traj.groupby("particle")["frame"].count()
    r_by_pid, rstar = free_curated_pids(out, T, min_len)
    # order by track length (most-sampled first); cap for a legible grid
    pids = [p for p in counts.sort_values(ascending=False).index
            if p in r_by_pid and counts[p] >= min_len]
    n_all = len(pids)
    capped = n_all > max_beads
    pids = pids[:max_beads]
    if not pids:
        print(f"[plot1_perbead] {stem}: no free curated beads -> skip")
        return None

    ncols = 4 if len(pids) > 6 else max(1, len(pids))
    nrows = int(np.ceil(len(pids) / ncols))
    figstyle.set_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False)
    plot_lag = min(max_lag, int(2 * fit_lag))

    n_flag_r2 = n_flag_drift = 0
    for k, pid in enumerate(pids):
        ax = axes[k // ncols][k % ncols]
        g = traj[traj["particle"] == pid].sort_values("frame")
        lag, msd_px2, npair, dxm, dym = msdmod.per_bead_msd(
            g["frame"].values, g["x"].values, g["y"].values, max_lag)
        fit = msdmod.fit_D(lag, msd_px2, npair, mpp, dt, fit_lag)
        if fit is None:
            ax.axis("off"); continue
        vx, vy = msdmod.fit_drift(lag, dxm, dym, mpp, dt, fit_lag)
        vmag = float(np.hypot(vx, vy))
        D, sD, r2 = fit["D_um2_s"], fit["D_err"], fit["r2_msd"]
        fa = g["frame"].values
        span = float((fa.max() - fa.min())) * dt
        sig_v = float(np.sqrt(2.0 * D / span)) if span > 0 else np.nan   # noise floor
        z = vmag / sig_v if (np.isfinite(sig_v) and sig_v > 0) else np.nan
        bad_r2 = np.isfinite(r2) and r2 < r2_thresh
        bad_drift = (np.isfinite(vmag) and vmag > drift_thresh
                     and np.isfinite(z) and z > 2.0)
        n_flag_r2 += int(bad_r2); n_flag_drift += int(bad_drift)

        tau = lag * dt
        msd_um2 = msd_px2 * mpp * mpp
        sel = lag <= plot_lag
        ax.plot(tau[sel], msd_um2[sel], "o", ms=3.2, color="#1f77b4", zorder=3)
        tg = np.linspace(0, plot_lag * dt, 40)
        ax.plot(tg, 4 * D * tg + fit["intercept_um2"],
                ls="--" if bad_r2 else "-", color="#d62728", lw=1.8, zorder=4)
        ax.set_xlim(0, plot_lag * dt); ax.set_ylim(0, None)
        ax.set_xlabel(r"$\tau$ [s]", fontsize=8)
        ax.set_ylabel(r"$\langle r^2\rangle$ [$\mu$m$^2$]", fontsize=8)
        ax.tick_params(labelsize=7)
        r2txt = rf"$R^2$={r2:.3f}" + ("  (low)" if bad_r2 else "")
        ax.text(0.04, 0.96,
                rf"$D$={D:.3f}$\pm${sD:.3f}" "\n" + r2txt + f"\n$r$={r_by_pid[pid]:.2f}$\\mu$m",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        tcol = "#d62728" if bad_drift else "0.25"
        ax.set_title(f"p{pid}  |v|={vmag*1e3:.0f}$\\pm${sig_v*1e3:.0f} nm/s"
                     + ("  drift!" if bad_drift else ""),
                     fontsize=8.5, color=tcol)

        # inset: per-bead drift check <Dx>,<Dy> vs tau + linear fits
        ins = inset_axes(ax, width="42%", height="34%", loc="lower right",
                         borderpad=0.6)
        s2 = lag <= plot_lag
        ins.axhline(0, color="0.8", lw=0.6)
        ins.plot(tau[s2], dxm[s2] * mpp, ".", ms=2.4, color="#1f77b4")
        ins.plot(tau[s2], dym[s2] * mpp, ".", ms=2.4, color="#2ca02c")
        ins.plot(tg, vx * tg, "-", color="#1f77b4", lw=1.0)
        ins.plot(tg, vy * tg, "-", color="#2ca02c", lw=1.0)
        ins.tick_params(labelsize=5.5, length=2)
        ins.set_title(r"$\langle\Delta x\rangle$(b), $\langle\Delta y\rangle$(g)",
                      fontsize=6)

    for k in range(len(pids), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    cap = f"  (showing {len(pids)} of {n_all} free beads)" if capped else ""
    fig.suptitle(
        f"{stem}: per-bead MSD + drift check  (T={T:.1f}$\\pm${T_unc:.0f}$^\\circ$C, "
        f"free diffusers $r\\leq r^*$={rstar:.2f}$\\mu$m){cap}\n"
        f"dashed fit = R$^2$<{r2_thresh};  red title = significant drift "
        f"(|v|>{drift_thresh*1e3:.0f} nm/s AND >2$\\sigma_v$)",
        fontsize=11, y=1.003)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    p = figstyle.save(fig, os.path.join(out, "plot1_perbead.png"))
    plt.close(fig)
    print(f"[plot1_perbead] {stem}: {len(pids)} beads "
          f"({n_flag_r2} R^2-flagged, {n_flag_drift} drift-flagged) -> {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description="Per-bead MSD + drift diagnostic grid.")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--max-beads", type=int, default=24)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-lag", type=int, default=100)
    ap.add_argument("--fit-lag", type=int, default=30)
    ap.add_argument("--drift-thresh", type=float, default=0.1)
    ap.add_argument("--r2-thresh", type=float, default=0.95)
    args = ap.parse_args()
    for stem in args.runs:
        run(stem, max_beads=args.max_beads, min_len=args.min_len,
            max_lag=args.max_lag, fit_lag=args.fit_lag,
            drift_thresh=args.drift_thresh, r2_thresh=args.r2_thresh)


if __name__ == "__main__":
    main()
