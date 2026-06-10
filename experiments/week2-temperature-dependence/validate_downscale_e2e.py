"""
validate_downscale_e2e.py
-------------------------
Decide whether half-res FRST detection (4.7x faster, but ~half the raw small-bead
detections) preserves the PHYSICS. Raw recall is the wrong metric -- we only keep
curated singles -- so we track week1 run3 at downscale=2 and compare its per-bead
D distribution + usable-track count to the committed FULL-RES run3 result. If the
median D and the number/size-span of long tracks survive, half-res is justified.

Run from the week root:  python validate_downscale_e2e.py
"""
import time
import numpy as np
import pandas as pd
from pipeline import frames as fr, track, msd

VID = "../week1-system-calibration/videos/run3.avi"
REF = "../week1-system-calibration/measurements/run3/pipeline/msd.csv"
MPP, FPS = 0.14381, 9.304


def d_dist(traj, min_len=60):
    dt = 1.0 / FPS
    Ds = []
    for _, g in traj.groupby("particle"):
        g = g.sort_values("frame")
        if len(g) < min_len:
            continue
        lag, m, npv = msd.per_bead_msd(g["frame"].values, g["x"].values,
                                       g["y"].values, 100)
        fit = msd.fit_D(lag, m, npv, MPP, dt, 30)
        if fit and np.isfinite(fit["D_um2_s"]):
            Ds.append(fit["D_um2_s"])
    return np.array(Ds)


def main():
    flat = fr.flat_field(VID, n_sample=60)
    t0 = time.time()
    traj, drift, jumps, (n0, n1) = track.track_clip(
        VID, flat=flat, detect_kw=dict(sym_min=0.18, grad_pct=80.0, downscale=2))
    secs = time.time() - t0
    D = d_dist(traj)
    ref = pd.read_csv(REF)
    print("\n================ HALF-RES vs FULL-RES (run3) ================")
    print(f"half-res track time : {secs/60:.1f} min for {n1} tracks "
          f"({secs/ max(traj['frame'].nunique(),1)*1000:.0f} ms/frame)")
    print(f"HALF-RES : {len(D):4d} beads >=60f   median D={np.median(D):.4f}  "
          f"IQR {np.percentile(D,25):.3f}-{np.percentile(D,75):.3f} um^2/s")
    print(f"FULL-RES : {len(ref):4d} beads        median D={ref['D_um2_s'].median():.4f}  "
          f"IQR {ref['D_um2_s'].quantile(.25):.3f}-{ref['D_um2_s'].quantile(.75):.3f} um^2/s "
          f"(committed)")
    print(f"median-D ratio half/full = {np.median(D)/ref['D_um2_s'].median():.3f}  "
          f"(want ~1.0 -> D preserved)")


if __name__ == "__main__":
    main()
