"""
contact_sheet.py  (week1-system-calibration)
--------------------------------------------
Make a labelling aid for Plot-2 curation: every tracked bead shown as one
clean, contrast-stretched crop (NO fit overlay, so sphericity is judged from
the raw image), tiled in a grid and tagged with its particle id. Also writes
labels_template.csv (particle, r_um, circ_resid_frac, + an empty `type` column)
for you to fill with sphere / doublet / blob.

Crops are sorted small -> large (by measured radius if radius.csv exists, else
by trackpy size from msd.csv) so the size progression is easy to scan.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/contact_sheet.py run3 --tag d21m600
    python scripts/contact_sheet.py run3
Then fill the `type` column of labels_template.csv (sphere/doublet/blob) and
save as labels.csv, OR just tell me the ids per category.
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


def gray_frame(cap, idx):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        return None
    fr = np.asarray(fr)
    if fr.ndim == 3:
        fr = fr[..., :3].mean(axis=-1)
    return fr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Contact sheet of bead crops for labelling.")
    ap.add_argument("run", help="run stem, e.g. run3")
    ap.add_argument("--tag", default=None, help="measurements/<run>/<tag>/")
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--half", type=int, default=22, help="crop half-width (px)")
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    import cv2

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    tcsv = os.path.join(cdir, "trajectory.csv")
    if not os.path.exists(tcsv):
        sys.exit(f"no trajectory.csv in {cdir} -- run track.py first")

    meta_video = _paths.load_runs().get("runs", {}).get(stem, {}).get("video", stem + ".avi")
    path = _paths.video(meta_video)
    traj = pd.read_csv(tcsv)
    counts = traj.groupby("particle")["frame"].count()
    pids = counts[counts >= args.min_len].index.tolist()

    # ordering + annotation context (radius if available, else trackpy size)
    rad = pd.read_csv(os.path.join(cdir, "radius.csv")) if os.path.exists(os.path.join(cdir, "radius.csv")) else None
    msd = pd.read_csv(os.path.join(cdir, "msd.csv")) if os.path.exists(os.path.join(cdir, "msd.csv")) else None
    info = {}
    for pid in pids:
        r_um = resid = np.nan
        if rad is not None and (rad.particle == pid).any():
            row = rad[rad.particle == pid].iloc[0]
            r_um, resid = row.get("r_um", np.nan), row.get("circ_resid_frac", np.nan)
        info[pid] = (r_um, resid)
    pids.sort(key=lambda p: (np.nan_to_num(info[p][0], nan=1e9), p))

    cap = cv2.VideoCapture(path)
    H = args.half
    crops = []
    for pid in pids:
        sub = traj[traj["particle"] == pid].sort_values("frame")
        mid = sub.iloc[len(sub) // 2]
        fr = gray_frame(cap, int(mid["frame"]))
        if fr is None:
            crops.append((pid, None)); continue
        x, y = float(mid["x"]), float(mid["y"])
        x0, y0 = int(round(x)) - H, int(round(y)) - H
        if x0 < 0 or y0 < 0 or x0 + 2 * H + 1 > fr.shape[1] or y0 + 2 * H + 1 > fr.shape[0]:
            crops.append((pid, None)); continue
        crop = fr[y0:y0 + 2 * H + 1, x0:x0 + 2 * H + 1]
        lo, hi = np.percentile(crop, [2, 98])
        crops.append((pid, np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1)))
    cap.release()

    cols = args.cols
    rows = int(np.ceil(len(crops) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.7 * rows), squeeze=False)
    for k, (pid, cr) in enumerate(crops):
        ax = axes[k // cols][k % cols]
        if cr is not None:
            ax.imshow(cr, cmap="gray")
        r_um, resid = info[pid]
        rtxt = f" r={r_um:.1f}\u00b5m" if np.isfinite(r_um) else ""
        ax.set_title(f"p{pid}{rtxt}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(crops), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{stem}: {len(crops)} beads (sorted small\u2192large). "
                 f"Label each: sphere / doublet / blob", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    sheet = os.path.join(cdir, "contact_sheet.png")
    fig.savefig(sheet, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[sheet] wrote {sheet} ({len(crops)} beads)")

    # labels template
    tmpl = pd.DataFrame({"particle": [p for p, _ in crops]})
    tmpl["r_um"] = [info[p][0] for p in tmpl.particle]
    tmpl["circ_resid_frac"] = [info[p][1] for p in tmpl.particle]
    tmpl["type"] = ""    # <-- fill: sphere / doublet / blob
    tpath = os.path.join(cdir, "labels_template.csv")
    tmpl.to_csv(tpath, index=False)
    print(f"[sheet] wrote {tpath}  (fill the 'type' column, save as labels.csv)")


if __name__ == "__main__":
    main()
