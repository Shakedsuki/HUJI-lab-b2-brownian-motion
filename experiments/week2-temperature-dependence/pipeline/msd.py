"""
msd.py  (pipeline)
------------------
trajectory.csv (drift-subtracted, PIXELS) -> per-bead D.

For each track we form the time-averaged 2D MSD
        MSD(tau) = < |r(t+tau) - r(t)|^2 >_t          [um^2]
(gaps from linker `memory` are handled: a lagged pair counts only if both frames
are present), convert to physical units with um/px from calibration/scale.json
and the clip's MEASURED dt = 1/fps from videos_meta.json, and fit the SHORT-lag
region
        MSD(tau) = 4 D tau + c .
The 4 is 2*(2 dims): in 2D, <r^2> = <dx^2>+<dy^2> = 2Dt + 2Dt = 4Dt. The intercept
c (>0) absorbs static localization noise + motion blur and MUST be fit, not
forced through 0, or D is biased. Only short lags are fit -- long lags droop from
the few independent samples in a finite track and would bias the slope.

Computed for every track with >= min_len frames; aggregate.py selects the
confirmed singles. Outputs msd.csv + Plot 1 (MSD overlay).
"""

import os
import numpy as np
import pandas as pd


def per_bead_msd(frame, x, y, max_lag):
    """Time-averaged MSD in px^2 vs lag (frames), gap-aware. Returns
    (lag_frames, msd_px2, n_pairs)."""
    frame = frame.astype(int)
    f0 = frame.min()
    L = int(frame.max() - f0 + 1)
    X = np.full(L, np.nan)
    Y = np.full(L, np.nan)
    X[frame - f0] = x
    Y[frame - f0] = y
    lags, msd, npair = [], [], []
    for tau in range(1, min(max_lag, L - 1) + 1):
        dx = X[tau:] - X[:-tau]
        dy = Y[tau:] - Y[:-tau]
        sd = dx * dx + dy * dy
        ok = ~np.isnan(sd)
        if ok.sum() < 1:
            continue
        lags.append(tau)
        msd.append(float(np.nanmean(sd)))
        npair.append(int(ok.sum()))
    return np.array(lags), np.array(msd), np.array(npair)


def fit_D(lag_frames, msd_px2, npair, mpp, dt, fit_lag):
    """Fit MSD = 4 D tau + c over lags <= fit_lag. Returns dict with D + errors."""
    lagt = lag_frames * dt
    msd_um2 = msd_px2 * mpp * mpp
    m = (lag_frames <= fit_lag) & np.isfinite(msd_um2)
    if m.sum() < 3:
        return None
    (slope, intercept), cov = np.polyfit(lagt[m], msd_um2[m], 1, cov=True)
    D = slope / 4.0
    D_err = float(np.sqrt(max(cov[0, 0], 0))) / 4.0
    # anomalous exponent over the same window (should be ~1 for free diffusion)
    pos = (lagt > 0) & (msd_um2 > 0) & m
    alpha = (np.polyfit(np.log(lagt[pos]), np.log(msd_um2[pos]), 1)[0]
             if pos.sum() >= 3 else np.nan)
    return dict(D_um2_s=float(D), D_err=float(D_err),
                intercept_um2=float(intercept), alpha=float(alpha),
                fit_npts=int(m.sum()))


def run(stem, min_len=60, max_lag=100, fit_lag=30, videos_dir=None):
    from . import paths, figstyle
    import matplotlib.pyplot as plt

    out = paths.out_dir(stem)
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    mpp = paths.load_scale() or 1.0
    fps = paths.fps_of(paths.video_for_run(stem)) or 9.30
    dt = 1.0 / fps
    print(f"[msd] {stem}: mpp={mpp} um/px, fps={fps:.3f} (dt={dt:.4f}s)")

    counts = traj.groupby("particle")["frame"].count()
    pids = counts[counts >= min_len].index.tolist()
    print(f"[msd] {len(pids)} tracks with >= {min_len} frames")

    rows, curves = [], {}
    for pid in pids:
        g = traj[traj["particle"] == pid].sort_values("frame")
        lag, msd_px2, npair = per_bead_msd(g["frame"].values, g["x"].values,
                                           g["y"].values, max_lag)
        if len(lag) < 3:
            continue
        fit = fit_D(lag, msd_px2, npair, mpp, dt, fit_lag)
        if fit is None:
            continue
        rows.append(dict(particle=int(pid), n_frames=int(counts[pid]), **fit))
        curves[pid] = (lag, msd_px2 * mpp * mpp, npair)   # lag(frames), MSD(um^2), n_pairs

    df = pd.DataFrame(rows).sort_values("n_frames", ascending=False)
    df.to_csv(os.path.join(out, "msd.csv"), index=False)
    print(f"[msd] wrote msd.csv ({len(df)} beads); "
          f"D range {df['D_um2_s'].min():.3f}-{df['D_um2_s'].max():.3f} um^2/s, "
          f"median alpha={df['alpha'].median():.2f}")

    # ---- ensemble MSD over the CLEAN (curated) set, for a simple linear figure
    from . import curate
    kept = curate.kept_pids(out)
    plot_pids = [p for p in curves if (kept is None or p in kept)] or list(curves)
    ens_sum, ens_w = {}, {}
    for p in plot_pids:                                  # pooled, weighted by n_pairs
        lag, M, NP = curves[p]
        for L, m, w in zip(lag, M, NP):
            ens_sum[L] = ens_sum.get(L, 0.0) + m * w
            ens_w[L] = ens_w.get(L, 0.0) + w
    Ls = np.array(sorted(ens_sum))
    ens = np.array([ens_sum[L] / ens_w[L] for L in Ls])
    ens_t = Ls * dt
    fm = Ls <= fit_lag
    slope, intercept = (np.polyfit(ens_t[fm], ens[fm], 1) if fm.sum() >= 2
                        else (np.nan, np.nan))
    D_ens = slope / 4.0

    figstyle.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lim = min(max_lag, int(2 * fit_lag))                 # linear panel range
    # LEFT: the clear one -- <r^2> linear in tau, ensemble + straight-line fit
    for p in plot_pids:
        lag, M, _ = curves[p]
        sel = lag <= lim
        ax[0].plot(lag[sel] * dt, M[sel], color="0.82", lw=0.5, alpha=0.4)
    es = Ls <= lim
    ax[0].plot(ens_t[es], ens[es], "o", color="C0", ms=5, zorder=3,
               label=r"ensemble $\langle r^2\rangle$")
    tg = np.linspace(0, lim * dt, 50)
    ax[0].plot(tg, slope * tg + intercept, "r-", lw=2.4, zorder=4,
               label=rf"fit $\langle r^2\rangle = 4D\tau + c$,  "
                     rf"$\langle D\rangle$ = {D_ens:.3f} $\mu$m$^2$/s")
    ax[0].set_xlim(0, lim * dt); ax[0].set_ylim(0, None)
    ax[0].set_xlabel(r"lag time $\tau$ [s]")
    ax[0].set_ylabel(r"$\langle r^2 \rangle$  [$\mu$m$^2$]")
    ax[0].set_title(f"{stem}: MSD is linear in time  (n={len(plot_pids)} singles)")
    ax[0].legend(loc="upper left", fontsize=9)
    # RIGHT: log-log diagnostic (normal diffusion => slope 1)
    for p in plot_pids:
        lag, M, _ = curves[p]
        ax[1].plot(lag * dt, M, color="0.78", lw=0.6, alpha=0.5)
    ax[1].plot(ens_t, ens, "C0-", lw=2, label=r"ensemble $\langle r^2\rangle$")
    ax[1].plot(ens_t, ens[0] / ens_t[0] * ens_t, "r--", lw=1.2,
               label="slope 1 (normal diffusion)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"lag time $\tau$ [s]")
    ax[1].set_ylabel(r"$\langle r^2 \rangle$  [$\mu$m$^2$]")
    ax[1].set_title("log-log check: exponent $\\approx$ 1"); ax[1].legend(fontsize=9)
    p = figstyle.save(fig, os.path.join(out, "plot1_msd.png"))
    plt.close(fig)
    print(f"[msd] wrote plot1_msd.png ({len(plot_pids)} singles; "
          f"ensemble <D>={D_ens:.3f} um^2/s) -> {out}")
    return df


if __name__ == "__main__":   # python -m pipeline.msd run3
    import argparse
    ap = argparse.ArgumentParser(description="Per-bead MSD -> D.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-lag", type=int, default=100)
    ap.add_argument("--fit-lag", type=int, default=30)
    args = ap.parse_args()
    run(args.run, min_len=args.min_len, max_lag=args.max_lag, fit_lag=args.fit_lag)
