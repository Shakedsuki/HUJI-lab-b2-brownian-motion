"""
validate.py  (pipeline)
-----------------------
Score the AUTOMATIC curation proposal against the human labels.csv from the OLD
pipeline (run3 has one: type in {perfect, singlet, doublet, blob}). This is a
ground-truth sanity check on PURITY, not a dependency -- the semi-automatic
workflow still gives the human the final say in this pipeline's own labels.csv.

The two pipelines assign different particle ids, so we match by POSITION: old
mean positions (from the old trajectory.csv) vs our curation x_med/y_med, after
removing any global coordinate offset (old positions are drift-subtracted; ours
are raw) via the median matched displacement.

Reports precision (of beads WE call single, how many the human called single ->
purity) and recall (of the human's singles, how many we keep), plus the
dangerous false positives (contaminants we'd have kept).
"""

import os
import numpy as np
import pandas as pd

from . import paths

SINGLE_TYPES = {"perfect", "singlet", "single"}


def run(stem, tol_px=16.0):
    from scipy.spatial import cKDTree

    out = paths.out_dir(stem)
    old = paths.old_dir(stem)
    cur = pd.read_csv(os.path.join(out, "curation.csv"))
    lab_path = os.path.join(old, "labels.csv")
    otraj_path = os.path.join(old, "trajectory.csv")
    if not (os.path.exists(lab_path) and os.path.exists(otraj_path)):
        print(f"[validate] missing old labels/trajectory for {stem} -- skipping")
        return None

    lab = pd.read_csv(lab_path)
    if "type" not in lab:
        print("[validate] old labels.csv has no 'type' column -- skipping")
        return None
    opos = pd.read_csv(otraj_path).groupby("particle")[["x", "y"]].mean()
    lab = lab.merge(opos, left_on="particle", right_index=True, how="inner")
    lab = lab.dropna(subset=["x", "y"])

    new = cur[["particle", "x_med", "y_med", "proposed", "reason"]].dropna(
        subset=["x_med", "y_med"]).reset_index(drop=True)
    newxy = new[["x_med", "y_med"]].to_numpy()
    oldxy = lab[["x", "y"]].to_numpy()

    # remove global frame offset (raw vs drift-subtracted) using coarse matches
    tree = cKDTree(newxy)
    d0, i0 = tree.query(oldxy, k=1)
    coarse = d0 < 40.0
    if coarse.sum() >= 3:
        off = np.median(oldxy[coarse] - newxy[i0[coarse]], axis=0)
    else:
        off = np.array([0.0, 0.0])
    d, idx = tree.query(oldxy - off, k=1)
    matched = d < tol_px

    lab = lab.copy()
    lab["human_single"] = lab["type"].isin(SINGLE_TYPES)
    lab["matched"] = matched
    lab["new_proposed"] = np.where(matched, new["proposed"].to_numpy()[idx], "unmatched")

    m = lab[lab["matched"]]
    TP = int(((m["human_single"]) & (m["new_proposed"] == "single")).sum())
    FP = int((~m["human_single"] & (m["new_proposed"] == "single")).sum())
    FN = int((m["human_single"] & (m["new_proposed"] == "reject")).sum())
    TN = int((~m["human_single"] & (m["new_proposed"] == "reject")).sum())
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")

    print(f"[validate] {stem}: matched {int(matched.sum())}/{len(lab)} labeled beads "
          f"(offset=({off[0]:.1f},{off[1]:.1f})px, tol={tol_px}px)")
    print(f"[validate]   human singles={int(m['human_single'].sum())}, "
          f"human contaminants={int((~m['human_single']).sum())}")
    print(f"[validate]   confusion: TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"[validate]   PRECISION (purity of our 'single' set) = {prec:.2f}")
    print(f"[validate]   RECALL    (fraction of human singles kept) = {rec:.2f}")
    if FP:
        fp = m[(~m["human_single"]) & (m["new_proposed"] == "single")]
        print(f"[validate]   !! {FP} FALSE POSITIVES (contaminants we'd keep): "
              + ", ".join(f"{t}@({x:.0f},{y:.0f})"
                          for t, x, y in zip(fp["type"], fp["x"], fp["y"])))
    unm = lab[~lab["matched"] & lab["human_single"]]
    if len(unm):
        print(f"[validate]   note: {len(unm)} human singles had no new track within tol "
              f"(short/lost or detector miss)")
    return dict(precision=prec, recall=rec, TP=TP, FP=FP, FN=FN, TN=TN)


if __name__ == "__main__":   # python -m pipeline.validate run3
    import argparse
    ap = argparse.ArgumentParser(description="Validate curation vs old labels.csv.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--tol-px", type=float, default=16.0)
    args = ap.parse_args()
    run(args.run, tol_px=args.tol_px)
