"""
curate_review.py <run>
----------------------
LOCAL interactive curation GUI (matplotlib window). Run on your machine after a
run is tracked + curated:

    python curate_review.py run7

A window shows every reviewable bead -- the auto-proposed singles (start green =
keep) and the borderline rejects (start red = reject) -- as a crop labelled with
its p-id, reject-reason, and shape metrics. Confident multi-reason rejects are
auto-excluded (not shown).

  click a bead .... toggle keep (green) <-> reject (red)
  a ............... all keep
  d ............... all reject
  s ............... SAVE -> writes labels.csv (kept beads) and closes
  q ............... quit without saving

labels.csv then drives MSD/radius (kept beads only). Re-run:
    python analyze_run.py run7
"""
import os
import sys

import numpy as np
import pandas as pd

from pipeline import paths, frames as fr

HARD = {"rigid-doublet", "two-cores", "mislink"}


def _classify(row):
    reasons = [r for r in str(row["reason"]).split(";") if r]
    if row["proposed"] == "single":
        return "keep"
    if len(reasons) == 1:
        return "drop" if reasons[0] in HARD else "borderline"
    return "drop"


def review(stem, cols=8):
    import cv2
    import matplotlib.pyplot as plt

    out = paths.out_dir(stem)
    cur = pd.read_csv(os.path.join(out, "curation.csv"))
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    vid = paths.video(paths.video_for_run(stem))
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"))

    cur["cls"] = cur.apply(_classify, axis=1)
    show = cur[cur["cls"].isin(["keep", "borderline"])].sort_values(
        ["cls", "sym_med"], ascending=[True, False])
    mid = {int(p): g.sort_values("frame").iloc[len(g) // 2]
           for p, g in traj.groupby("particle")}

    cap = cv2.VideoCapture(vid)
    crops = []
    for _, r in show.iterrows():
        pid = int(r["particle"])
        if pid not in mid:
            continue
        m = mid[pid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(m["frame"]))
        ok, frm = cap.read()
        if not ok:
            continue
        img = frm[..., :3].mean(-1).astype(np.float32) - flat
        h = int(max(2.4 * r["R_px_med"], 20))
        H, W = img.shape
        x, y = int(m["x_raw"]), int(m["y_raw"])
        c = img[max(0, y - h):min(H, y + h), max(0, x - h):min(W, x + h)]
        crops.append((pid, c, r["cls"], round(float(r["sym_med"]), 2),
                      round(float(r["ecc_med"]), 2),
                      (str(r["reason"]) or "single")))
    cap.release()
    if not crops:
        print(f"[curate_review] {stem}: nothing to review")
        return

    n = len(crops)
    rows = int(np.ceil(n / cols))
    state = {pid: (cls == "keep") for pid, _, cls, _, _, _ in crops}
    fig, axs = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.9 * rows))
    axs = np.array(axs).reshape(rows, cols)
    axmap = {}

    def paint(ax, pid):
        col = "#23c552" if state[pid] else "#e8482e"
        for s in ax.spines.values():
            s.set_color(col)
            s.set_linewidth(3)

    for k, (pid, c, cls, sym, ecc, reason) in enumerate(crops):
        ax = axs[k // cols][k % cols]
        ax.imshow(c, cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"p{pid} {reason}\nsym{sym} ecc{ecc}", fontsize=6.5)
        paint(ax, pid)
        axmap[ax] = pid
    for k in range(n, rows * cols):
        axs[k // cols][k % cols].axis("off")

    def save():
        df = pd.DataFrame({"particle": sorted(state),
                           "keep": [int(state[p]) for p in sorted(state)]})
        df.to_csv(os.path.join(out, "labels.csv"), index=False)
        nk = int(df["keep"].sum())
        print(f"[curate_review] {stem}: wrote labels.csv -- {nk}/{len(df)} kept "
              f"-> {out}\n  now: python analyze_run.py {stem}")

    def on_click(ev):
        if ev.inaxes in axmap:
            pid = axmap[ev.inaxes]
            state[pid] = not state[pid]
            paint(ev.inaxes, pid)
            fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key in ("a", "d"):
            for pid in state:
                state[pid] = (ev.key == "a")
            for ax, pid in axmap.items():
                paint(ax, pid)
            fig.canvas.draw_idle()
        elif ev.key == "s":
            save()
            plt.close(fig)
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.suptitle(f"{stem}: click=toggle keep(green)/reject(red)   "
                 f"a=all-keep  d=all-reject  s=SAVE+labels.csv  q=quit  "
                 f"({n} beads)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    review(sys.argv[1] if len(sys.argv) > 1 else "run7")
