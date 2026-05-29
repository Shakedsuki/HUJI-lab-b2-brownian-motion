"""
radius.py  (pipeline)
---------------------
Physical radius per confirmed single, from the OUTER EDGE of the dark
diffraction ring (shape.measure_shape), median over ~15 frames.

THE DIFFRACTION BIAS (stated, not silently corrected): the ring's outer edge
sits a roughly CONSTANT few-tenths-of-a-um OUTSIDE the true sphere edge, because
diffraction spreads the ring. A constant-px offset biases SMALL beads
proportionally more (it's a larger fraction of their radius), exactly where D is
also noisiest -- so it is the accuracy bottleneck for k_B. We report the raw
outer-edge radius + reliability flags here, and aggregate.py propagates a
+/- delta_px radius-bias band into the k_B error budget rather than guessing a
single correction.

radius.csv columns: r_px_med, r_um, R_cv (focus stability), resid_med,
ring_cv_med, ecc_med, inlier_med, nn_px_med (isolation), n_shape.
"""

import os
import numpy as np
import pandas as pd

from . import shape


def isolation(traj):
    """Per-track median nearest-neighbour distance (px) over its frames. Small =>
    a close companion that can corrupt the edge fit (crowding / resolved doublet)."""
    from scipy.spatial import cKDTree
    nn = {}
    for _, g in traj.groupby("frame"):
        if len(g) < 2:
            continue
        pts = g[["x_raw", "y_raw"]].to_numpy()
        d, _ = cKDTree(pts).query(pts, k=2)
        for pid, dd in zip(g["particle"].to_numpy(), d[:, 1]):
            nn.setdefault(int(pid), []).append(float(dd))
    return {pid: float(np.median(v)) for pid, v in nn.items()}


def _montage(df, traj, video, flat, path, mpp, cols=8):
    import cv2
    import matplotlib.pyplot as plt
    from . import figstyle
    n = len(df)
    if n == 0:
        return None
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.0 * rows),
                            squeeze=False)
    th = np.linspace(0, 2 * np.pi, 90)
    cap = cv2.VideoCapture(video)
    for k, (_, r) in enumerate(df.iterrows()):
        pid = int(r["particle"])
        g = traj[traj["particle"] == pid].sort_values("frame")
        row = g.iloc[len(g) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, fr = cap.read()
        ax = axs[k // cols][k % cols]
        ax.axis("off")
        if not ok:
            continue
        img = fr[..., :3].mean(-1).astype(np.float32) - flat
        m = shape.measure_shape(img, float(row["x_raw"]), float(row["y_raw"]),
                                float(row["r_est"]), int(row["polarity"]))
        ax.imshow(m["_crop"], cmap="gray")
        if np.isfinite(m["R"]):
            cxy = m["_cxy"]
            ax.plot(cxy[0] + m["R"] * np.cos(th), cxy[1] + m["R"] * np.sin(th),
                    "r-", lw=1.0)
        ax.set_title(f"p{pid} r={r['r_um']:.2f}um", fontsize=6.5)
    for k in range(n, rows * cols):
        axs[k // cols][k % cols].axis("off")
    cap.release()
    fig.suptitle("radius_check: outer-edge circle fit on confirmed singles")
    p = figstyle.save(fig, path, dpi=130)
    plt.close(fig)
    return p


def run(stem, n_global=120, min_len=60, videos_dir=None):
    from . import paths, frames as fr, curate

    out = paths.out_dir(stem)
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    mpp = paths.load_scale() or 1.0
    counts = traj.groupby("particle")["frame"].count()

    kept = curate.kept_pids(out)
    if kept is None:
        kept = set(counts[counts >= min_len].index)
        print(f"[radius] no labels/proposal -> measuring all {len(kept)} long tracks")
    pids = [int(p) for p in kept if counts.get(p, 0) >= min(min_len, 30)]
    print(f"[radius] {stem}: measuring radius for {len(pids)} beads, mpp={mpp} um/px")

    vid = paths.video(paths.video_for_run(stem), videos_dir)
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"), n_sample=60)
    traj_s = traj.sort_values(["particle", "frame"])
    keep_set = set(pids)
    groups = {int(p): g for p, g in traj_s.groupby("particle") if int(p) in keep_set}
    sub = traj_s[traj_s["particle"].isin(keep_set)]
    frame_groups = {int(f): g for f, g in sub.groupby("frame")}
    agg = curate.measure_tracks(vid, flat, groups, frame_groups, n_global=n_global)
    nn = isolation(traj)

    rows = []
    for pid in pids:
        a = agg.get(pid)
        if a is None:
            continue
        rows.append(dict(
            particle=int(pid),
            r_px_med=a["R_px_med"], r_um=a["R_px_med"] * mpp,
            R_cv=a["R_cv"], resid_med=a["resid_med"],
            ring_cv_med=a["ring_cv_med"], ecc_med=a["ecc_med"],
            inlier_med=a["inlier_med"], nn_px_med=nn.get(pid, np.nan),
            n_shape=a["n_shape"]))
    df = pd.DataFrame(rows).sort_values("r_um")
    df.to_csv(os.path.join(out, "radius.csv"), index=False)
    print(f"[radius] wrote radius.csv ({len(df)} beads); "
          f"r_um range {df['r_um'].min():.2f}-{df['r_um'].max():.2f} um")
    p = _montage(df, traj, vid, flat, os.path.join(out, "radius_check.png"), mpp)
    if p:
        print(f"[radius] wrote radius_check.png -> {out}")
    return df


if __name__ == "__main__":   # python -m pipeline.radius run3
    import argparse
    ap = argparse.ArgumentParser(description="Physical radius via outer-edge fit.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--n-global", type=int, default=120)
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()
    run(args.run, n_global=args.n_global, videos_dir=args.videos_dir)
