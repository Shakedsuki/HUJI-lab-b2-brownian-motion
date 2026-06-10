"""
contact_sheet.py  (pipeline)
----------------------------
The human's confirmation surface for the semi-automatic curation. Renders every
track as a crop (its median frame) + the fitted circle + a score line + a
proposed verdict, sorted by beadness, split into the PROPOSED-SINGLES sheet (to
confirm/veto) and the REJECTED sheet (to scan for anything wrongly tossed). Also
emits per-criterion histograms with the gate cut lines, and a labels.csv
template the human edits (keep = 1/0) -- downstream uses labels.csv if present.
"""

import os
import numpy as np
import pandas as pd

from . import shape, figstyle


def _rep_frame(traj, pid):
    g = traj[traj["particle"] == pid].sort_values("frame")
    row = g.iloc[len(g) // 2]
    return int(row["frame"]), float(row["x_raw"]), float(row["y_raw"]), \
        float(row["r_est"]), int(row["polarity"])


def _grid(df, traj, video, flat, path, title, cols=8, cell=1.9):
    import cv2
    import matplotlib.pyplot as plt
    n = len(df)
    if n == 0:
        return None
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows),
                            squeeze=False)
    th = np.linspace(0, 2 * np.pi, 90)
    # read each needed frame once
    want = {}
    meta = {}
    for _, r in df.iterrows():
        pid = int(r["particle"])
        f, x, y, re, pol = _rep_frame(traj, pid)
        want.setdefault(f, []).append(pid)
        meta[pid] = (f, x, y, re, pol, r)
    crops = {}
    cap = cv2.VideoCapture(video)
    for f in sorted(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        img = fr[..., :3].mean(-1).astype(np.float32)
        if flat is not None:
            img = img - flat
        for pid in want[f]:
            _, x, y, re, pol, _ = meta[pid]
            crops[pid] = (img, x, y, re, pol)
    cap.release()

    for k, (_, r) in enumerate(df.iterrows()):
        pid = int(r["particle"])
        ax = axs[k // cols][k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if pid not in crops:
            ax.axis("off"); continue
        img, x, y, re, pol = crops[pid]
        m = shape.measure_shape(img, x, y, re, pol)
        cr = m["_crop"]
        ax.imshow(cr, cmap="gray")
        cxy = m["_cxy"]
        if np.isfinite(m["R"]):
            ax.plot(cxy[0] + m["R"] * np.cos(th), cxy[1] + m["R"] * np.sin(th),
                    "-", color="orange", lw=1.0)
        good = (r["proposed"] == "single")
        for sp in ax.spines.values():
            sp.set_color("limegreen" if good else "red")
            sp.set_linewidth(2.2)
        lbl = f"p{pid} sym{r['sym_med']:.0f}\nres{r['resid_med']:.02f} " \
              f"cv{r['ring_cv_med']:.02f} ec{r['ecc_med']:.02f}"
        if not good:
            lbl += f"\n{r['reason'][:18]}"
        ax.set_title(lbl, fontsize=6.0)
    for k in range(n, rows * cols):
        axs[k // cols][k % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    p = figstyle.save(fig, path, dpi=130)
    plt.close(fig)
    return p


def _histograms(df, path, gates):
    import matplotlib.pyplot as plt
    specs = [("resid_med", gates["resid"], "circle resid/R"),
             ("ring_cv_med", gates["ring_cv"], "ring CV"),
             ("ecc_med", gates["ecc"], "edge eccentricity"),
             ("R_cv", gates["rcv"], "radius CV (focus)"),
             ("n_frames", gates["min_len"], "track length"),
             ("R_px_med", None, "radius (px)")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    keep = df["proposed"] == "single"
    for ax, (col, cut, lab) in zip(axs.ravel(), specs):
        v = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            ax.hist(df.loc[keep, col].dropna(), bins=20, alpha=0.7,
                    label="proposed single", color="C2")
            ax.hist(df.loc[~keep, col].dropna(), bins=20, alpha=0.6,
                    label="rejected", color="C3")
        if cut is not None:
            ax.axvline(cut, color="k", ls="--", lw=1.2, label=f"gate={cut}")
        ax.set_xlabel(lab); ax.set_ylabel("tracks"); ax.legend(fontsize=7)
    fig.suptitle("Curation criteria (green kept / red rejected; dashed = gate)")
    p = figstyle.save(fig, path, dpi=130)
    plt.close(fig)
    return p


def write_labels_template(df, out, overwrite=False):
    """labels.csv the human edits: keep = 1/0 (prefilled from the proposal)."""
    path = os.path.join(out, "labels.csv")
    if os.path.exists(path) and not overwrite:
        print(f"[sheet] labels.csv already exists -> left as-is ({path})")
        return path
    t = df[["particle", "proposed", "reason", "R_px_med", "sym_med"]].copy()
    t["keep"] = (df["proposed"] == "single").astype(int)
    t = t[["particle", "keep", "proposed", "reason", "R_px_med", "sym_med"]]
    t.to_csv(path, index=False)
    print(f"[sheet] wrote labels.csv template ({t['keep'].sum()} keep) -> {path}")
    return path


def render(stem, videos_dir=None, n_flat=60, max_frames=None, gates=None):
    from . import paths, frames as fr, curate

    gates = gates or curate.GATES
    out = paths.out_dir(stem)
    df = pd.read_csv(os.path.join(out, "curation.csv"))
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    vid = paths.video(paths.video_for_run(stem), videos_dir)
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"),
                       n_sample=n_flat, max_frames=max_frames)

    singles = df[df["proposed"] == "single"].sort_values("sym_med", ascending=False)
    rejects = df[df["proposed"] == "reject"].sort_values("sym_med", ascending=False)
    p1 = _grid(singles, traj, vid, flat, os.path.join(out, "sheet_singles.png"),
               f"{stem}: PROPOSED SINGLES (n={len(singles)}) -- confirm/veto in labels.csv")
    p2 = _grid(rejects, traj, vid, flat, os.path.join(out, "sheet_rejected.png"),
               f"{stem}: REJECTED (n={len(rejects)}) -- scan for good beads to rescue")
    p3 = _histograms(df, os.path.join(out, "curation_hist.png"), gates)
    write_labels_template(df, out)
    print(f"[sheet] wrote {p1}")
    print(f"[sheet] wrote {p2}")
    print(f"[sheet] wrote {p3}")


if __name__ == "__main__":   # python -m pipeline.contact_sheet run3
    import argparse
    ap = argparse.ArgumentParser(description="Render the curation contact sheet.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()
    render(args.run, videos_dir=args.videos_dir)
