"""
review.py  (pipeline)
---------------------
Human-verification aids on top of curation.csv, to make manual eyeballing
efficient. All read existing outputs (no re-tracking). Four views:

  borderline  : ONLY the tracks whose scores sit near a gate -> the ambiguous
                ~dozen where your judgement actually flips the outcome.
  field       : the whole frame with every track centre marked green (proposed
                single) / red (reject) + particle id -> spatial scan for a missed
                bead or a kept contaminant.
  strips      : per-track row of crops sampled across the clip -> see the bead
                move through focus; catch defocus / doublets that only separate
                in some frames.
  vs_old      : each track annotated with the OLD pipeline's human label at the
                matched position, disagreements first -> where I differ from you.

Usage:
    python -m pipeline.review run3 --aid all
    python -m pipeline.review run3 --aid borderline
"""

import os
import numpy as np
import pandas as pd

from . import paths, figstyle

GATES_REF = [("resid_med", 0.10), ("ring_cv_med", 0.10),
             ("ecc_med", 0.45), ("R_cv", 0.25)]
SINGLE_TYPES = {"perfect", "singlet", "single"}


def _load(stem):
    out = paths.out_dir(stem)
    cur = pd.read_csv(os.path.join(out, "curation.csv"))
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    vid = paths.video(paths.video_for_run(stem))
    return out, cur, traj, vid


def _read_crops(vid, flat, reqs, half=22):
    """reqs: list of (key, frame, x, y). Returns {key: (crop, cx, cy)}; reads each
    frame once."""
    import cv2
    by_frame = {}
    for key, f, x, y in reqs:
        by_frame.setdefault(int(f), []).append((key, x, y))
    crops = {}
    cap = cv2.VideoCapture(vid)
    H = W = None
    for f in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        img = fr[..., :3].mean(-1).astype(np.float32)
        if flat is not None:
            img = img - flat
        H, W = img.shape
        for key, x, y in by_frame[f]:
            xi, yi = int(round(x)), int(round(y))
            a, b = max(0, xi - half), max(0, yi - half)
            crops[key] = (img[b:min(H, yi + half), a:min(W, xi + half)],
                          x - a, y - b)
    cap.release()
    return crops


def _rep(traj, pid):
    g = traj[traj["particle"] == pid].sort_values("frame")
    r = g.iloc[len(g) // 2]
    return int(r["frame"]), float(r["x_raw"]), float(r["y_raw"]), \
        float(r["r_est"]), int(r["polarity"])


def match_new_to_old(stem, cur, tol_px=16.0):
    """For each new track (curation row), the OLD pipeline's human label type, or
    'unlabeled'. Matches by position with a global offset removed."""
    from scipy.spatial import cKDTree
    old = paths.old_dir(stem)
    lp, tp = os.path.join(old, "labels.csv"), os.path.join(old, "trajectory.csv")
    types = np.array(["unlabeled"] * len(cur), dtype=object)
    if not (os.path.exists(lp) and os.path.exists(tp)):
        return types
    lab = pd.read_csv(lp)
    if "type" not in lab:
        return types
    opos = pd.read_csv(tp).groupby("particle")[["x", "y"]].mean()
    lab = lab.merge(opos, left_on="particle", right_index=True, how="inner").dropna(
        subset=["x", "y"])
    oldxy = lab[["x", "y"]].to_numpy()
    newxy = cur[["x_med", "y_med"]].to_numpy()
    tree = cKDTree(oldxy)
    d0, i0 = tree.query(newxy, k=1)
    coarse = d0 < 40.0
    off = (np.median(newxy[coarse] - oldxy[i0[coarse]], axis=0)
           if coarse.sum() >= 3 else np.array([0.0, 0.0]))
    d, idx = tree.query(newxy - off, k=1)
    ok = d < tol_px
    types[ok] = lab["type"].to_numpy()[idx[ok]]
    return types


def _montage(rows, crops, path, title, cols=8, cell=1.9):
    """rows: list of dict(key, title, good[bool], R, cxy). Generic crop grid."""
    import matplotlib.pyplot as plt
    n = len(rows)
    if n == 0:
        print(f"[review] {title}: nothing to show")
        return None
    nr = int(np.ceil(n / cols))
    fig, axs = plt.subplots(nr, cols, figsize=(cell * cols, cell * nr), squeeze=False)
    th = np.linspace(0, 2 * np.pi, 90)
    for k, r in enumerate(rows):
        ax = axs[k // cols][k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        cc = crops.get(r["key"])
        if cc is None:
            ax.axis("off"); continue
        crop, cx, cy = cc
        ax.imshow(crop, cmap="gray")
        if r.get("R") and np.isfinite(r["R"]):
            ax.plot(cx + r["R"] * np.cos(th), cy + r["R"] * np.sin(th),
                    "-", color="orange", lw=1.0)
        for sp in ax.spines.values():
            sp.set_color("limegreen" if r["good"] else "red")
            sp.set_linewidth(2.2)
        ax.set_title(r["title"], fontsize=6.0)
    for k in range(n, nr * cols):
        axs[k // cols][k % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    p = figstyle.save(fig, path, dpi=130)
    plt.close(fig)
    return p


def borderline(stem, max_cells=32, flat=None):
    out, cur, traj, vid = _load(stem)
    if flat is None:
        from . import frames as fr
        flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"), n_sample=60)
    md = np.full(len(cur), np.inf)
    worst = np.array(["" for _ in range(len(cur))], dtype=object)
    for col, g in GATES_REF:
        rel = np.abs(cur[col] - g) / g
        upd = rel < md
        md = np.where(upd, rel, md)
        worst = np.where(upd, col, worst)
    cur = cur.assign(flip_dist=md, worst=worst)
    # the most ambiguous tracks = smallest distance-to-a-gate; cap so the human
    # only reviews the genuinely borderline ~couple-dozen, not the whole field.
    bl = cur.sort_values("flip_dist").head(max_cells)
    reqs, rows = [], []
    for _, r in bl.iterrows():
        pid = int(r["particle"])
        f, x, y, re, pol = _rep(traj, pid)
        reqs.append((pid, f, x, y))
        rows.append(dict(key=pid, good=(r["proposed"] == "single"), R=None,
                         title=f"p{pid} {r['proposed']}\n{r['worst'][:8]} "
                               f"Δ{r['flip_dist']:.0%}\n{str(r['reason'])[:16]}"))
    crops = _read_crops(vid, flat, reqs)
    p = _montage(rows, crops, os.path.join(out, "review_borderline.png"),
                 f"{stem}: BORDERLINE tracks (near a gate; Δ = rel. dist to flip)")
    print(f"[review] borderline: {len(bl)} tracks -> {p}")
    return p


def field(stem, frame_idx=0, flat=None):
    import cv2
    import matplotlib.pyplot as plt
    out, cur, traj, vid = _load(stem)
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read(); cap.release()
    img = fr[..., :3].mean(-1)
    figstyle.set_style()
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img, cmap="gray")
    for _, r in cur.iterrows():
        good = r["proposed"] == "single"
        c = "limegreen" if good else "red"
        ax.plot(r["x_med"], r["y_med"], "o", mfc="none", mec=c, ms=10, mew=1.3)
        ax.text(r["x_med"] + 6, r["y_med"], str(int(r["particle"])), color=c,
                fontsize=6)
    ns = int((cur["proposed"] == "single").sum())
    ax.set_title(f"{stem}: field map (green=proposed single n={ns}, red=reject "
                 f"n={len(cur)-ns}); id labels")
    ax.axis("off")
    p = figstyle.save(fig, os.path.join(out, "review_field.png"), dpi=150)
    plt.close(fig)
    print(f"[review] field overlay -> {p}")
    return p


def strips(stem, which="single", n_t=7, max_rows=60, flat=None):
    import matplotlib.pyplot as plt
    out, cur, traj, vid = _load(stem)
    if flat is None:
        from . import frames as fr
        flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"), n_sample=60)
    sub = cur[cur["proposed"] == ("single" if which == "single" else "reject")]
    sub = sub.sort_values("sym_med", ascending=False).head(max_rows)
    reqs = []
    plan = {}
    for _, r in sub.iterrows():
        pid = int(r["particle"])
        g = traj[traj["particle"] == pid].sort_values("frame")
        idxs = np.linspace(0, len(g) - 1, n_t).astype(int)
        fs = []
        for j in idxs:
            row = g.iloc[j]
            reqs.append(((pid, int(row["frame"])), int(row["frame"]),
                         float(row["x_raw"]), float(row["y_raw"])))
            fs.append((pid, int(row["frame"])))
        plan[pid] = fs
    crops = _read_crops(vid, flat, reqs, half=20)
    nr = len(plan)
    if nr == 0:
        print(f"[review] strips ({which}): none"); return None
    fig, axs = plt.subplots(nr, n_t, figsize=(1.3 * n_t, 1.3 * nr), squeeze=False)
    for i, (pid, fs) in enumerate(plan.items()):
        for j, key in enumerate(fs):
            ax = axs[i][j]; ax.set_xticks([]); ax.set_yticks([])
            cc = crops.get(key)
            if cc is not None:
                ax.imshow(cc[0], cmap="gray")
            if j == 0:
                ax.set_ylabel(f"p{pid}", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle(f"{stem}: time-strips ({which}s, frames start->end) -- watch for "
                 f"defocus / a 2nd core appearing", fontsize=10)
    p = figstyle.save(fig, os.path.join(out, f"review_strips_{which}.png"), dpi=130)
    plt.close(fig)
    print(f"[review] strips ({which}): {nr} tracks -> {p}")
    return p


def vs_old(stem, flat=None):
    out, cur, traj, vid = _load(stem)
    if flat is None:
        from . import frames as fr
        flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"), n_sample=60)
    old_type = match_new_to_old(stem, cur)
    cur = cur.assign(old_type=old_type)
    cur = cur[cur["old_type"] != "unlabeled"].copy()
    if len(cur) == 0:
        print("[review] vs_old: no matches to old labels"); return None
    cur["old_single"] = cur["old_type"].isin(SINGLE_TYPES)
    cur["mine_single"] = cur["proposed"] == "single"
    cur["disagree"] = cur["old_single"] != cur["mine_single"]
    cur = cur.sort_values(["disagree", "sym_med"], ascending=[False, False])
    reqs, rows = [], []
    for _, r in cur.iterrows():
        pid = int(r["particle"])
        f, x, y, re, pol = _rep(traj, pid)
        reqs.append((pid, f, x, y))
        flag = "  <DIFF" if r["disagree"] else ""
        rows.append(dict(key=pid, good=not r["disagree"], R=None,
                         title=f"p{pid}\nmine:{r['proposed'][:3]} old:{r['old_type'][:4]}{flag}"))
    crops = _read_crops(vid, flat, reqs)
    nd = int(cur["disagree"].sum())
    p = _montage(rows, crops, os.path.join(out, "review_vs_old.png"),
                 f"{stem}: vs OLD labels ({nd} disagreements first; green=agree)")
    print(f"[review] vs_old: {len(cur)} matched, {nd} disagreements -> {p}")
    return p


def render_all(stem):
    from . import frames as fr
    vid = paths.video(paths.video_for_run(stem))
    flat = fr.get_flat(vid, cache_path=os.path.join(paths.out_dir(stem), "flat.npy"),
                       n_sample=60)
    borderline(stem, flat=flat)
    field(stem, flat=flat)
    strips(stem, which="single", flat=flat)
    strips(stem, which="reject", flat=flat)
    vs_old(stem, flat=flat)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Manual-verification aids.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--aid", choices=["all", "borderline", "field", "strips", "vs_old"],
                    default="all")
    args = ap.parse_args()
    if args.aid == "all":
        render_all(args.run)
    else:
        globals()[args.aid](args.run)
