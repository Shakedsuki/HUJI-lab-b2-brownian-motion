"""
msd_fit.py  (week1-system-calibration)
--------------------------------------
trajectory.csv  ->  per-bead MSD, per-bead D, and the diagnostics we curate on.

For each particle (bead) it computes the time-averaged MSD via trackpy.imsd
(this IS the single-bead analysis, done for every bead at once), converts to
physical units using calibration/scale.json (um/px) and the clip's measured fps,
fits the short-lag region

        <r^2>(tau) = 4 D tau + c           [um^2] = [um^2/s][s]

so D = slope/4 (2D projection; the factor is 4, not 1), and records:
  - D_um2_s, D_err        : the bead's diffusion coefficient + fit error
  - intercept_um2         : static-localisation / blur offset (should be small)
  - n_frames              : track length (long => reliable D)
  - size_px_med, size_cv  : trackpy apparent size + its variability
                            (size_cv large => bead defocuses in/out -> bad r)
  - ecc_med               : eccentricity (high => merged/contact doublet)
  - nn_px_med             : median nearest-neighbour distance over the track
                            (small => persistent close companion = resolved
                            doublet / crowded -> curate out)
  - mass_med              : brightness (helps flag doublets/aggregates)

Outputs measurements/<stem>/msd.csv (+ an MSD overlay = the booklet's Plot 1).
The D-vs-(1/r) plot (Plot 2) is built later by pooling curated beads across the
room runs; this script produces the per-bead inputs for it.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/msd_fit.py run2 --min-len 100 --max-lag 100 --fit-lag 30
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths
import figure_style


def main():
    ap = argparse.ArgumentParser(description="Per-bead MSD and D from trajectory.csv.")
    ap.add_argument("run", help="run stem, e.g. run2")
    ap.add_argument("--min-len", type=int, default=100, help="drop beads with fewer frames")
    ap.add_argument("--max-lag", type=int, default=100, help="MSD computed up to this lag (frames)")
    ap.add_argument("--fit-lag", type=int, default=30, help="fit the slope over lags <= this (frames)")
    ap.add_argument("--tag", default=None,
                    help="read/write measurements/<run>/<tag>/ instead of "
                         "measurements/<run>/ (match the tag used in track.py).")
    args = ap.parse_args()

    import trackpy as tp

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    csv = os.path.join(cdir, "trajectory.csv")
    if not os.path.exists(csv):
        sys.exit(f"no trajectory.csv in {cdir} -- run track.py first")

    mpp = _paths.load_scale()
    if mpp is None:
        print("[warn] no um_per_px in scale.json; using 1.0 (results in PIXELS)")
        mpp = 1.0
    # measured fps for this clip's video, from videos_meta.json
    meta_video = _paths.load_runs().get("runs", {}).get(stem, {}).get("video", stem + ".avi")
    fps = _paths.fps_of(meta_video) or 9.30
    print(f"[msd] {stem}: mpp={mpp} um/px, fps={fps:.3f} (dt={1/fps:.4f} s)")

    traj = pd.read_csv(csv)
    # keep well-sampled beads
    counts = traj.groupby("particle")["frame"].count()
    keep = counts[counts >= args.min_len].index
    traj = traj[traj["particle"].isin(keep)]
    print(f"[msd] {len(keep)} beads with >= {args.min_len} frames")
    if len(keep) == 0:
        sys.exit("no beads long enough; lower --min-len or check tracking")

    # per-bead time-averaged MSD (um^2 vs lag time in s)
    im = tp.imsd(traj, mpp, fps, max_lagtime=args.max_lag)
    lagt = im.index.values                      # seconds
    fit_mask = lagt <= args.fit_lag / fps       # short-lag fit window

    # isolation: per-bead median nearest-neighbour distance (px) across its
    # frames. A small value => a persistent close companion (resolved doublet
    # or crowded neighbourhood) that ecc misses -> a curation handle for Plot 2.
    from scipy.spatial import cKDTree
    nn_accum = {}
    for _, g in traj.groupby("frame"):
        if len(g) < 2:
            continue
        pts = g[["x", "y"]].values
        dist, _ = cKDTree(pts).query(pts, k=2)
        for pid_, dd in zip(g["particle"].values, dist[:, 1]):
            nn_accum.setdefault(pid_, []).append(float(dd))
    nn_med = {pid_: float(np.median(v)) for pid_, v in nn_accum.items()}

    rows = []
    for pid in im.columns:
        y = im[pid].values
        m = fit_mask & np.isfinite(y)
        if m.sum() < 3:
            continue
        (slope, intercept), cov = np.polyfit(lagt[m], y[m], 1, cov=True)
        slope_err = float(np.sqrt(cov[0, 0]))
        sub = traj[traj["particle"] == pid]
        size_med = float(sub["size"].median()) if "size" in sub else np.nan
        size_cv = float(sub["size"].std() / sub["size"].mean()) if "size" in sub else np.nan
        rows.append({
            "particle": int(pid),
            "n_frames": int(len(sub)),
            "D_um2_s": slope / 4.0,
            "D_err": slope_err / 4.0,
            "intercept_um2": intercept,
            "fit_npts": int(m.sum()),
            "size_px_med": size_med,
            "size_cv": size_cv,
            "ecc_med": float(sub["ecc"].median()) if "ecc" in sub else np.nan,
            "nn_px_med": nn_med.get(pid, np.nan),
            "mass_med": float(sub["mass"].median()) if "mass" in sub else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("n_frames", ascending=False)
    out.to_csv(os.path.join(cdir, "msd.csv"), index=False)
    print(f"[msd] wrote msd.csv ({len(out)} beads) -> {cdir}")

    # ensemble D for a quick single number
    em = tp.emsd(traj, mpp, fps, max_lagtime=args.max_lag)
    emask = em.index.values <= args.fit_lag / fps
    es = np.polyfit(em.index.values[emask], em.values[emask], 1)[0]
    print(f"[msd] ensemble D ~= {es/4:.4f} um^2/s   "
          f"(per-bead D range {out['D_um2_s'].min():.3f}-{out['D_um2_s'].max():.3f})")

    # Plot 1: per-bead MSD overlay + ensemble, log-log, with a slope-1 guide
    figure_style.set_style()
    plt.figure(figsize=(6.4, 5))
    for pid in im.columns:
        plt.plot(im.index, im[pid], color="0.7", lw=0.7, alpha=0.6)
    plt.plot(em.index, em.values, "k-", lw=2.2, label="ensemble  <r^2>")
    g = em.index.values
    plt.plot(g, em.values[0] / g[0] * g, "r--", lw=1.2, label="slope 1 (normal diffusion)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("lag time  tau  [s]"); plt.ylabel(r"$\langle r^2\rangle$  [$\mu$m$^2$]")
    plt.title(f"{stem}: per-bead MSD ({len(out)} beads)")
    plt.legend()
    p = os.path.join(cdir, "msd_overlay.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[msd] wrote msd_overlay.png -> {cdir}")
    print("      curate msd.csv (long n_frames, small size_cv, resolvable size) "
          "for the D-vs-1/r plot.")


if __name__ == "__main__":
    main()
