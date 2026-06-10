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
    """Time-averaged MSD in px^2 vs lag (frames), gap-aware. Also returns the
    mean SIGNED displacement per lag (for the per-bead drift check). Returns
    (lag_frames, msd_px2, n_pairs, dx_mean_px, dy_mean_px)."""
    frame = frame.astype(int)
    f0 = frame.min()
    L = int(frame.max() - f0 + 1)
    X = np.full(L, np.nan)
    Y = np.full(L, np.nan)
    X[frame - f0] = x
    Y[frame - f0] = y
    lags, msd, npair, dxm, dym = [], [], [], [], []
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
        dxm.append(float(np.nanmean(dx)))      # <Dx(tau)> -> drift slope
        dym.append(float(np.nanmean(dy)))
    return (np.array(lags), np.array(msd), np.array(npair),
            np.array(dxm), np.array(dym))


def _r2(y, yhat):
    """Coefficient of determination of a fit (1 - SS_res/SS_tot)."""
    y = np.asarray(y, float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - yhat) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_D(lag_frames, msd_px2, npair, mpp, dt, fit_lag):
    """Fit MSD = 4 D tau + c over lags <= fit_lag. Returns dict with D, its
    fit-covariance error sigma_D = sigma_slope/4, and the R^2 of the linear fit
    (MSD linearity metric)."""
    lagt = lag_frames * dt
    msd_um2 = msd_px2 * mpp * mpp
    m = (lag_frames <= fit_lag) & np.isfinite(msd_um2)
    if m.sum() < 3:
        return None
    (slope, intercept), cov = np.polyfit(lagt[m], msd_um2[m], 1, cov=True)
    D = slope / 4.0
    D_err = float(np.sqrt(max(cov[0, 0], 0))) / 4.0
    r2 = _r2(msd_um2[m], slope * lagt[m] + intercept)
    # anomalous exponent over the same window (should be ~1 for free diffusion)
    pos = (lagt > 0) & (msd_um2 > 0) & m
    alpha = (np.polyfit(np.log(lagt[pos]), np.log(msd_um2[pos]), 1)[0]
             if pos.sum() >= 3 else np.nan)
    return dict(D_um2_s=float(D), D_err=float(D_err),
                intercept_um2=float(intercept), alpha=float(alpha),
                r2_msd=float(r2), fit_npts=int(m.sum()))


def fit_drift(lag_frames, dx_mean_px, dy_mean_px, mpp, dt, fit_lag):
    """Per-bead drift velocity from <Dx(tau)>, <Dy(tau)> vs tau (linear fit;
    slope = v). On the drift-SUBTRACTED coordinates this is the RESIDUAL drift
    that survived the global de-drift -> a clean flag for beads with local
    convection / wall-sliding / mislinks. Returns (v_dx, v_dy) in um/s."""
    lagt = lag_frames * dt
    m = (lag_frames <= fit_lag) & np.isfinite(dx_mean_px) & np.isfinite(dy_mean_px)
    if m.sum() < 3:
        return np.nan, np.nan
    vx = float(np.polyfit(lagt[m], dx_mean_px[m] * mpp, 1)[0])   # um/s
    vy = float(np.polyfit(lagt[m], dy_mean_px[m] * mpp, 1)[0])
    return vx, vy


def run(stem, min_len=60, max_lag=100, fit_lag=30, videos_dir=None,
        drift_thresh=0.1, r2_thresh=0.95):
    from . import paths, figstyle
    import matplotlib.pyplot as plt

    out = paths.out_dir(stem)
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    mpp = paths.load_scale() or 1.0
    fps = paths.fps_of(paths.video_for_run(stem)) or 9.30
    dt = 1.0 / fps
    print(f"[msd] {stem}: mpp={mpp} um/px, fps={fps:.3f} (dt={dt:.4f}s); "
          f"flags: |v_drift|>{drift_thresh} um/s, R^2<{r2_thresh}")

    counts = traj.groupby("particle")["frame"].count()
    pids = counts[counts >= min_len].index.tolist()
    print(f"[msd] {len(pids)} tracks with >= {min_len} frames")

    rows, curves = [], {}
    for pid in pids:
        g = traj[traj["particle"] == pid].sort_values("frame")
        lag, msd_px2, npair, dxm, dym = per_bead_msd(
            g["frame"].values, g["x"].values, g["y"].values, max_lag)
        if len(lag) < 3:
            continue
        fit = fit_D(lag, msd_px2, npair, mpp, dt, fit_lag)
        if fit is None:
            continue
        vx, vy = fit_drift(lag, dxm, dym, mpp, dt, fit_lag)
        vmag = float(np.hypot(vx, vy)) if np.isfinite(vx) else np.nan
        # statistical drift-velocity noise floor: the MLE drift over a track of
        # duration T_span has Var = 2D/T_span (net displacement / time), so a
        # finite track shows |v|~sqrt(2D/T) even with ZERO true drift. Flag drift
        # only when it clears BOTH the physical threshold AND ~2 sigma_v (real,
        # not finite-track noise).
        fr_arr = g["frame"].values
        span = float((fr_arr.max() - fr_arr.min())) * dt
        sig_v = float(np.sqrt(2.0 * fit["D_um2_s"] / span)) if span > 0 else np.nan
        z = vmag / sig_v if (np.isfinite(sig_v) and sig_v > 0) else np.nan
        drift_flag = bool(np.isfinite(vmag) and vmag > drift_thresh
                          and np.isfinite(z) and z > 2.0)
        row = dict(particle=int(pid), n_frames=int(counts[pid]), **fit,
                   v_dx_um_s=vx, v_dy_um_s=vy, v_drift_um_s=vmag,
                   v_drift_se=sig_v, drift_z=z, drift_flag=drift_flag,
                   lin_flag=bool(np.isfinite(fit["r2_msd"])
                                 and fit["r2_msd"] < r2_thresh))
        rows.append(row)
        # lag(frames), MSD(um^2), n_pairs, <Dx>(um), <Dy>(um) -- for per-bead figs
        curves[pid] = (lag, msd_px2 * mpp * mpp, npair, dxm * mpp, dym * mpp)

    df = pd.DataFrame(rows).sort_values("n_frames", ascending=False)
    df.to_csv(os.path.join(out, "msd.csv"), index=False)
    n_drift = int(df["drift_flag"].sum()) if len(df) else 0
    n_lin = int(df["lin_flag"].sum()) if len(df) else 0
    print(f"[msd] wrote msd.csv ({len(df)} beads); "
          f"D range {df['D_um2_s'].min():.3f}-{df['D_um2_s'].max():.3f} um^2/s, "
          f"median alpha={df['alpha'].median():.2f}, median R^2={df['r2_msd'].median():.3f}")
    df[["particle", "n_frames", "v_dx_um_s", "v_dy_um_s", "v_drift_um_s",
        "v_drift_se", "drift_z", "drift_flag", "r2_msd", "lin_flag"]].to_csv(
        os.path.join(out, "drift_perbead.csv"), index=False)
    print(f"[msd] wrote drift_perbead.csv; {n_drift} bead(s) flagged drift "
          f"(|v|>{drift_thresh} um/s AND >2 sigma_v), {n_lin} bead(s) R^2<{r2_thresh}")

    # ---- ensemble MSD over the CLEAN (curated) set, for a simple linear figure
    from . import curate
    kept = curate.kept_pids(out)
    plot_pids = [p for p in curves if (kept is None or p in kept)] or list(curves)
    ens_sum, ens_w = {}, {}
    for p in plot_pids:                                  # pooled, weighted by n_pairs
        lag, M, NP, _dx, _dy = curves[p]
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
        lag, M, _NP, _dx, _dy = curves[p]
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
        lag, M, _NP, _dx, _dy = curves[p]
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
    ap.add_argument("--drift-thresh", type=float, default=0.1,
                    help="flag beads with residual |v_drift| > this [um/s]")
    ap.add_argument("--r2-thresh", type=float, default=0.95,
                    help="flag beads whose MSD linear-fit R^2 < this")
    args = ap.parse_args()
    run(args.run, min_len=args.min_len, max_lag=args.max_lag, fit_lag=args.fit_lag,
        drift_thresh=args.drift_thresh, r2_thresh=args.r2_thresh)
