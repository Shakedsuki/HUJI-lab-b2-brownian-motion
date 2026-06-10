"""
seed_match.py <run>
-------------------
Connect reference-frame seeds (seeds.csv) to the tracked beads. For each seed,
find the track passing nearest its (x, y) at the seed frame -> that particle is a
trusted bead carrying the human radius. Writes:
  labels.csv         (particle, keep=1)         -- the seeded set
  radius_manual.csv  (particle, r_um_manual...) -- human radii

analyze_run.py then uses both automatically. Run AFTER tracking.

Usage:  python seed_match.py run7 [--tol 8]
"""
import os
import sys

import numpy as np
import pandas as pd

from pipeline import paths


def match(stem, tol=8.0):
    out = paths.out_dir(stem)
    seeds = pd.read_csv(os.path.join(out, "seeds.csv"))
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))

    rows, miss = [], []
    for _, s in seeds.iterrows():
        at = traj[traj["frame"] == int(s["frame"])]
        if not len(at):
            miss.append((s["x"], s["y"], "no detections at seed frame"))
            continue
        dist = np.hypot(at["x_raw"].to_numpy() - s["x"],
                        at["y_raw"].to_numpy() - s["y"])
        k = int(np.argmin(dist))
        if dist[k] <= max(tol, float(s["r_px"])):
            rows.append(dict(particle=int(at.iloc[k]["particle"]),
                             r_um_manual=float(s["r_um"]),
                             r_px_manual=float(s["r_px"]),
                             match_px=round(float(dist[k]), 2)))
        else:
            miss.append((s["x"], s["y"], f"nearest track {dist[k]:.1f}px away"))

    man = pd.DataFrame(rows).drop_duplicates("particle")
    man.to_csv(os.path.join(out, "radius_manual.csv"), index=False)
    pd.DataFrame({"particle": man["particle"], "keep": 1}).to_csv(
        os.path.join(out, "labels.csv"), index=False)
    print(f"[seed_match] {stem}: {len(man)} seeds matched to tracks, "
          f"{len(miss)} unmatched (of {len(seeds)})")
    if miss:
        print("  unmatched seeds (bead may not have tracked >=60 frames): "
              + "; ".join(f"({x:.0f},{y:.0f}) {why}" for x, y, why in miss[:6])
              + (" ..." if len(miss) > 6 else ""))
    print(f"  wrote labels.csv + radius_manual.csv -> {out}")
    print(f"  now: python analyze_run.py {stem}")
    return man


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Match seeds to tracked beads.")
    ap.add_argument("run", nargs="?", default="run7")
    ap.add_argument("--tol", type=float, default=8.0)
    args = ap.parse_args()
    match(args.run, tol=args.tol)
