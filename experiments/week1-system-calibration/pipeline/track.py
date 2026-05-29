"""
track.py  (pipeline)
--------------------
FRST detections -> linked, drift-subtracted trajectories.

Linking is the one part of the classic stack that works well, so we keep
trackpy (Crocker-Grier nearest-neighbour with memory) for it. Detection is ours
(detect.py); curation is ours (curate.py).

Drift: a collective flow (stage creep, slow leak, a table bump) would otherwise
masquerade as super-diffusion -- <r^2> picks up a v^2 t^2 term that biases D
high. We estimate the per-frame collective drift as the MEDIAN of all particles'
frame-to-frame displacements (median, not trackpy's mean, so a few fast/mislinked
beads don't pull it) and subtract its cumulative sum. We also flag sudden jumps
(the booklet's ~40 px table bump) as change-points to report.

trajectory.csv keeps BOTH coordinate sets, all in PIXELS:
  x, y          drift-subtracted -> used by MSD/D (diffusion only)
  x_raw, y_raw  original detections -> used to crop the bead in the video for
                radius + the contact sheet (the video pixels live at the raw
                positions, not the de-drifted ones)
"""

import numpy as np
import pandas as pd

from . import detect


def robust_drift(traj, jump_k=6.0, jump_abs=8.0):
    """Cumulative collective drift (px) per frame + flagged sudden-jump frames.

    Returns (drift_df indexed by frame with columns x,y; list of (frame, mag))."""
    t = traj.sort_values(["particle", "frame"])
    dx = t.groupby("particle")["x"].diff()
    dy = t.groupby("particle")["y"].diff()
    dfn = t.groupby("particle")["frame"].diff()
    step = pd.DataFrame({"frame": t["frame"], "dx": dx, "dy": dy, "df": dfn})
    step = step[step["df"] == 1]                       # consecutive frames only
    inc = step.groupby("frame")[["dx", "dy"]].median()  # per-frame median increment

    mag = np.hypot(inc["dx"], inc["dy"])
    med = float(np.median(mag)) if len(mag) else 0.0
    mad = float(np.median(np.abs(mag - med))) * 1.4826 + 1e-9
    thr = max(med + jump_k * mad, jump_abs)
    jumps = [(int(f), float(m)) for f, m in mag.items() if m > thr]

    fmin, fmax = int(traj["frame"].min()), int(traj["frame"].max())
    drift = inc.cumsum().reindex(range(fmin, fmax + 1)).interpolate().fillna(0.0)
    drift.index.name = "frame"
    return drift, jumps


def subtract_drift(traj, drift):
    out = traj.copy()
    out["x_raw"] = out["x"]
    out["y_raw"] = out["y"]
    out["x"] = out["x"] - out["frame"].map(drift["dx"]).to_numpy()
    out["y"] = out["y"] - out["frame"].map(drift["dy"]).to_numpy()
    return out


def track_clip(video_path, flat=None, search=8, memory=3, stub=50,
               max_frames=None, detect_kw=None, progress=100):
    """video -> drift-subtracted trajectory DataFrame, drift, jumps, counts."""
    import trackpy as tp
    tp.quiet()
    detect_kw = detect_kw or {}
    print(f"[track] detecting (FRST) over frames...")
    feats = detect.detect_clip(video_path, flat=flat, max_frames=max_frames,
                               progress=progress, **detect_kw)
    print(f"[track] {len(feats)} detections over {feats['frame'].nunique()} frames; linking...")
    # beads move only ~2-3 px/frame, so a small search_range is both correct and
    # necessary: too large -> dense subnets blow up (SubnetOversizeException).
    # adaptive_stop lets trackpy shrink the range locally in crowded spots.
    traj = tp.link(feats, search_range=search, memory=memory,
                   adaptive_stop=3.0, adaptive_step=0.9)
    n0 = traj["particle"].nunique()
    traj = tp.filter_stubs(traj, stub).reset_index(drop=True)   # 'frame' stays a column only
    n1 = traj["particle"].nunique()
    print(f"[track] {n0} tracks -> {n1} after dropping stubs < {stub} frames")
    drift, jumps = robust_drift(traj)
    traj_nd = subtract_drift(traj, drift)
    maxd = float(np.hypot(drift["dx"], drift["dy"]).max()) if len(drift) else 0.0
    print(f"[track] cumulative drift max |d| = {maxd:.1f} px; "
          f"{len(jumps)} sudden-jump frame(s)" + (f": {jumps[:5]}" if jumps else ""))
    return traj_nd, drift, jumps, (n0, n1)


def run(stem, videos_dir=None, search=8, memory=3, stub=50, max_frames=None,
        sym_min=0.18, grad_pct=80.0, n_flat=60):
    """Full per-clip tracking -> writes trajectory.csv, drift.csv, previews."""
    import os
    import matplotlib.pyplot as plt
    from . import paths, frames as fr, figstyle

    figstyle.set_style()
    vid = paths.video(paths.video_for_run(stem), videos_dir)
    out = paths.out_dir(stem)
    print(f"[track] {stem}: {vid}")
    print(f"[track] building flat-field ({n_flat} frames)...")
    flat = fr.flat_field(vid, n_sample=n_flat, max_frames=max_frames)

    traj, drift, jumps, (n0, n1) = track_clip(
        vid, flat=flat, search=search, memory=memory, stub=stub,
        max_frames=max_frames, detect_kw=dict(sym_min=sym_min, grad_pct=grad_pct))

    cols = ["particle", "frame", "x", "y", "x_raw", "y_raw", "sym", "r_est",
            "polarity", "contrast"]
    traj[cols].to_csv(os.path.join(out, "trajectory.csv"), index=False)
    drift.to_csv(os.path.join(out, "drift.csv"))
    pd.DataFrame(jumps, columns=["frame", "mag_px"]).to_csv(
        os.path.join(out, "drift_jumps.csv"), index=False)
    print(f"[track] wrote trajectory.csv ({n1} tracks) -> {out}")

    # previews: drift curve + trajectory map
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(drift.index, drift["dx"], label="drift x")
    ax[0].plot(drift.index, drift["dy"], label="drift y")
    for f, _ in jumps:
        ax[0].axvline(f, color="r", ls=":", lw=1)
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("cumulative drift (px)")
    ax[0].set_title(f"{stem}: collective drift (subtracted); "
                    f"{len(jumps)} jump(s) flagged"); ax[0].legend()
    for pid, g in traj.groupby("particle"):
        ax[1].plot(g["x"], g["y"], lw=0.5, alpha=0.6)
    ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    ax[1].set_xlabel("x (px, de-drifted)"); ax[1].set_ylabel("y (px)")
    ax[1].set_title(f"{n1} drift-subtracted trajectories")
    p = figstyle.save(fig, os.path.join(out, "track_overview.png"))
    plt.close(fig)
    print(f"[track] wrote track_overview.png -> {out}")
    return traj


if __name__ == "__main__":   # python -m pipeline.track run3 [--max-frames 100]
    import argparse
    ap = argparse.ArgumentParser(description="Track a clip (FRST + trackpy link).")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--search", type=float, default=8)
    ap.add_argument("--memory", type=int, default=3)
    ap.add_argument("--stub", type=int, default=50)
    ap.add_argument("--sym-min", type=float, default=0.18)
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()
    run(args.run, videos_dir=args.videos_dir, search=args.search,
        memory=args.memory, stub=args.stub, max_frames=args.max_frames,
        sym_min=args.sym_min)
