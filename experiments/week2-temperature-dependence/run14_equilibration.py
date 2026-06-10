"""
run14_equilibration.py
----------------------
run14 is the longest clip (4409 frames / 474 s); the standard pipeline caps
tracking at 1500 frames, sampling only the first ~161 s -- where D is still
rising. Here we track the FULL clip (no cap) and measure the ensemble diffusion
coefficient vs time: a direct view of the sample reaching thermal steady state.

Radius-free trend. We report median per-bead D and median D*r_auto in fixed time
windows. r_auto is the FRST/shape size; its diffraction over-read is CONSTANT in
time, so the TREND of D*r is unbiased even though its absolute level is not. To
kill bead-composition drift between windows we restrict to a fixed auto-size band.
We mark where the committed 1500-frame cut ends and overlay run13 (clean same-T,
24.3 C) and the Stokes-Einstein D*r expected at the 24.3 C label.

Does NOT touch the committed run14 outputs (writes only figures/).

Usage:  python run14_equilibration.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline import paths, physics, figstyle, track, curate
from pipeline import frames as fr
from pipeline import msd as M

STEM = "run14"
T_NOM = 24.3
R_BAND_UM = (0.85, 1.40)      # auto-size band -> fixed composition across windows
WIN_S = 40.0                  # time-window width [s]
MIN_PTS = 40                  # min frames a bead needs in a window to fit D
KB = physics.K_B


def window_D(traj, pids_r, mpp, dt, f0, f1):
    """Median per-bead D and D*r_auto over [f0,f1) for size-banded beads."""
    Ds, DRs = [], []
    for pid, r_au in pids_r.items():
        g = traj[(traj["particle"] == pid) & (traj["frame"] >= f0)
                 & (traj["frame"] < f1)].sort_values("frame")
        if len(g) < MIN_PTS:
            continue
        lag, m, npair, dx, dy = M.per_bead_msd(g["frame"].values, g["x"].values,
                                               g["y"].values, 60)
        fit = M.fit_D(lag, m, npair, mpp, dt, 20)
        if fit and np.isfinite(fit["D_um2_s"]) and fit["D_um2_s"] > 0:
            Ds.append(fit["D_um2_s"]); DRs.append(fit["D_um2_s"] * r_au)
    n = len(Ds)
    if n < 3:
        return None
    return dict(n=n, D=float(np.median(Ds)), Dr=float(np.median(DRs)),
                Dr_se=float(np.median(np.abs(np.array(DRs) - np.median(DRs)))
                            * 1.4826 / np.sqrt(n)))


def main():
    mpp = paths.load_scale() or 0.14381
    vid = paths.video(paths.video_for_run(STEM))
    fps = paths.fps_of(paths.video_for_run(STEM)) or 9.30
    dt = 1.0 / fps
    scratch = os.path.join(paths.out_dir(STEM), "equil")
    os.makedirs(scratch, exist_ok=True)
    print(f"[equil] {STEM}: full-clip track (no frame cap), mpp={mpp}, dt={dt:.4f}s",
          flush=True)
    flat = fr.get_flat(vid, cache_path=os.path.join(paths.out_dir(STEM), "flat.npy"),
                       n_sample=60)

    traj, drift, jumps, (n0, n1) = track.track_clip(
        vid, flat=flat, search=8, memory=3, stub=50, max_frames=None,
        detect_kw=dict(sym_min=0.18, grad_pct=80.0, downscale=2,
                       workers=max(1, (os.cpu_count() or 2) - 2)))
    traj.to_csv(os.path.join(scratch, "trajectory_full.csv"), index=False)
    fmax = int(traj["frame"].max())
    print(f"[equil] tracked {n1} tracks over {fmax + 1} frames "
          f"({(fmax + 1) * dt:.0f} s)", flush=True)

    # auto size per track (median outer-edge R), to band-limit composition
    groups = {int(p): g for p, g in traj.groupby("particle")}
    frame_groups = {int(f): g for f, g in traj.groupby("frame")}
    agg = curate.measure_tracks(vid, flat, groups, frame_groups, n_global=100)
    pids_r = {p: a["R_px_med"] * mpp for p, a in agg.items()
              if a is not None and R_BAND_UM[0] <= a["R_px_med"] * mpp <= R_BAND_UM[1]}
    print(f"[equil] {len(pids_r)} beads in auto-size band {R_BAND_UM} um", flush=True)

    win = int(round(WIN_S / dt))
    rows = []
    for f0 in range(0, fmax + 1, win):
        w = window_D(traj, pids_r, mpp, dt, f0, f0 + win)
        if w:
            rows.append(dict(t_mid=(f0 + win / 2) * dt, f0=f0, **w))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(paths.FIGURES_DIR, "run14_equilibration.csv"), index=False)
    print("[equil] windowed D(t):", flush=True)
    print(tab[["t_mid", "n", "D", "Dr"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    # SE-expected D*r at the 24.3 C label (for the band-median radius)
    rbar = float(np.median(list(pids_r.values())))
    Dr_se_pred = KB * (T_NOM + 273.15) / (6 * np.pi * physics.water_viscosity_Pa_s(T_NOM)) * 1e18

    figstyle.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax[0].plot(tab["t_mid"], tab["D"], "o-", color="C0", lw=1.8)
    ax[0].axvline(1500 * dt, color="C3", ls=":", lw=1.6,
                  label=f"committed cut (1500 fr = {1500*dt:.0f} s)")
    ax[0].set_xlabel("time into clip [s]"); ax[0].set_ylabel(r"median $D$ [$\mu$m$^2$/s]")
    ax[0].set_title(f"{STEM}: ensemble D vs time (size-banded {R_BAND_UM[0]}-{R_BAND_UM[1]} um)")
    ax[0].set_ylim(0, None); ax[0].legend(fontsize=9)

    ax[1].errorbar(tab["t_mid"], tab["Dr"], yerr=tab["Dr_se"], fmt="o-", color="C0",
                   lw=1.8, capsize=3, label=r"median $D\,r_{\rm auto}$ (trend; offset-biased level)")
    ax[1].axvline(1500 * dt, color="C3", ls=":", lw=1.6, label="committed cut")
    ax[1].set_xlabel("time into clip [s]"); ax[1].set_ylabel(r"$D\,r_{\rm auto}$ [$\mu$m$^3$/s]")
    ax[1].set_title("radius-free trend: sample reaching steady state?")
    ax[1].set_ylim(0, None); ax[1].legend(fontsize=8)
    p = figstyle.save(fig, os.path.join(paths.FIGURES_DIR, "run14_equilibration.png"))
    plt.close(fig)
    print(f"[equil] wrote {p}", flush=True)
    d0, d1 = tab["D"].iloc[0], tab["D"].iloc[-1]
    print(f"[equil] D start->end: {d0:.3f} -> {d1:.3f} um2/s "
          f"({(d1/d0-1)*100:+.0f}%); plateau check in the figure.", flush=True)


if __name__ == "__main__":
    main()
