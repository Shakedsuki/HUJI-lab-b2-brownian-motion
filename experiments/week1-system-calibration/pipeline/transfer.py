"""
transfer.py  (pipeline)
-----------------------
Carry human tags made on one curation (e.g. the early 120-frame quicklook in
early_tag/) onto a different track set (the full-clip curation), since the two
pipelines/runs assign different particle ids. Matching is by POSITION at a shared
frame: a tagged bead, at its own middle frame, is the same physical bead at the
same raw (x,y) in both track sets, so the nearest full-clip detection at that
frame is its counterpart (near-exact, tol ~5 px).

Writes the full curation's labels.csv merging: your verdict where a bead matches,
the auto proposal elsewhere (a `source` column marks which). Note: fragmentation
means one tagged bead maps to the full fragment covering its middle frame; other
fragments of the same bead fall back to auto until you review them.

    python -m pipeline.transfer run3                 # from early_tag/ -> full
"""

import os
import numpy as np
import pandas as pd

from . import paths


def run(stem, from_dir=None, tol_px=5.0):
    out = paths.out_dir(stem)
    src = from_dir or os.path.join(out, "early_tag")
    slab = pd.read_csv(os.path.join(src, "labels.csv"))
    straj = pd.read_csv(os.path.join(src, "trajectory.csv"))
    ftraj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    cur = pd.read_csv(os.path.join(out, "curation.csv"))

    sgroups = {int(p): g for p, g in straj.groupby("particle")}
    fbyframe = {int(f): g[["particle", "x_raw", "y_raw"]]
                for f, g in ftraj.groupby("frame")}

    transfer = {}                                       # full_pid -> verdict dict
    n_match = 0
    for _, r in slab.iterrows():
        g = sgroups.get(int(r["particle"]))
        if g is None or len(g) == 0:
            continue
        gm = g.sort_values("frame").iloc[len(g) // 2]
        fsub = fbyframe.get(int(gm["frame"]))
        if fsub is None or len(fsub) == 0:
            continue
        d = np.hypot(fsub["x_raw"].to_numpy() - float(gm["x_raw"]),
                     fsub["y_raw"].to_numpy() - float(gm["y_raw"]))
        k = int(np.argmin(d))
        if d[k] <= tol_px:
            transfer[int(fsub.iloc[k]["particle"])] = dict(
                keep=int(r["keep"]), type=str(r.get("type", "")),
                note=("" if pd.isna(r.get("note", "")) else str(r.get("note", ""))))
            n_match += 1

    rows = []
    for _, c in cur.iterrows():
        fp = int(c["particle"])
        if fp in transfer:
            t = transfer[fp]
            keep, source = t["keep"], "human"
            typ = t["type"] or ("single" if keep else "reject")
            note = t["note"]
        else:
            keep, source, note = int(c["proposed"] == "single"), "auto", ""
            typ = "single" if keep else "reject"
        rows.append(dict(particle=fp, keep=keep, type=typ, note=note, source=source,
                         proposed=c["proposed"], reason=c["reason"],
                         x_med=c["x_med"], y_med=c["y_med"],
                         R_px_med=c["R_px_med"], sym_med=c["sym_med"]))
    lab = pd.DataFrame(rows)
    lab.to_csv(os.path.join(out, "labels.csv"), index=False)

    hk = int(((lab.source == "human") & (lab.keep == 1)).sum())
    ak = int(((lab.source == "auto") & (lab.keep == 1)).sum())
    print(f"[transfer] matched {n_match}/{len(slab)} subset tags onto full tracks "
          f"(tol {tol_px}px)")
    print(f"[transfer] wrote full labels.csv: {int((lab.keep == 1).sum())} keep "
          f"({hk} from your tags, {ak} auto-proposed) of {len(lab)} tracks -> {out}")
    return lab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Transfer subset tags onto full tracks.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--from-dir", default=None)
    ap.add_argument("--tol-px", type=float, default=5.0)
    args = ap.parse_args()
    run(args.run, from_dir=args.from_dir, tol_px=args.tol_px)
