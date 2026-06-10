"""
radius_tag.py <run>
-------------------
Manual radius tagging GUI. The automatic radius reads the OUTER edge of the dark
diffraction ring, which over-reads the true sphere (worst for small beads) and
biases k_B high. Here you size a circle to each bead's true edge by eye ->
ground-truth radii that bypass the diffraction bias.

One bead at a time, shown at its SHARPEST frame (max FRST sym). Fit the circle to
your best estimate of the PHYSICAL sphere edge -- be consistent (same visual
feature every time, e.g. the bright core boundary).

  mouse click ...... move circle CENTER
  scroll up/down ... radius +/- 0.5 px
  up / down arrow .. radius +/- 0.1 px (fine)
  n / enter / right  accept radius, next bead
  b / left ......... previous bead
  x ................ toggle bead BAD (exclude from analysis)
  s ................ SAVE -> radius_manual.csv  (then closes)
  q ................ quit without saving

Then:  python analyze_run.py <run>      (prefers radius_manual.csv when present)
"""
import os
import sys

import numpy as np
import pandas as pd

from pipeline import paths, frames as fr, curate


def _beadlist(out):
    """Curated singles (labels.csv keep==1 if present, else auto proposal),
    smallest first so the size trend is easy to scan."""
    cur = pd.read_csv(os.path.join(out, "curation.csv"))
    kept = curate.kept_pids(out)
    cur = (cur[cur["particle"].isin(kept)] if kept is not None
           else cur[cur["proposed"] == "single"])
    return cur.sort_values("R_px_med")


def review(stem):
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    out = paths.out_dir(stem)
    mpp = paths.load_scale() or 0.14381
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    cur = _beadlist(out)
    vid = paths.video(paths.video_for_run(stem))
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"))

    beads = []
    cap = cv2.VideoCapture(vid)
    for _, r in cur.iterrows():
        pid = int(r["particle"])
        g = traj[traj["particle"] == pid]
        if not len(g):
            continue
        row = g.loc[g["sym"].idxmax()]                  # sharpest frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frm = cap.read()
        if not ok:
            continue
        img = frm[..., :3].mean(-1).astype(np.float32) - flat
        r_est = float(r["R_px_med"])
        h = int(max(4 * r_est, 40))
        H, W = img.shape
        x, y = float(row["x_raw"]), float(row["y_raw"])
        xi, yi = int(round(x)), int(round(y))
        a, b = max(0, yi - h), max(0, xi - h)
        crop = img[a:min(H, yi + h), b:min(W, xi + h)]
        beads.append(dict(pid=pid, crop=crop, cx=x - b, cy=y - a,
                          r=r_est, r_auto=r_est, bad=False, tagged=False))
    cap.release()
    if not beads:
        print(f"[radius_tag] {stem}: no beads to tag")
        return

    state = {"i": 0}
    fig, ax = plt.subplots(figsize=(8, 8.5))

    def show():
        bd = beads[state["i"]]
        ax.clear()
        ax.imshow(bd["crop"], cmap="gray")
        ax.add_patch(Circle((bd["cx"], bd["cy"]), bd["r"], fill=False,
                            edgecolor="#ff3030", lw=1.5))
        ax.plot(bd["cx"], bd["cy"], "+", color="#ff3030", ms=8)
        ax.set_xticks([]); ax.set_yticks([])
        tag = "BAD" if bd["bad"] else ("tagged" if bd["tagged"] else "untagged")
        ntag = sum(b["tagged"] and not b["bad"] for b in beads)
        ax.set_title(
            f"bead {state['i'] + 1}/{len(beads)}   p{bd['pid']}   [{tag}]   "
            f"({ntag} tagged)\n"
            f"r = {bd['r']:.1f} px = {bd['r'] * mpp:.3f} um"
            f"   (auto {bd['r_auto'] * mpp:.3f} um)\n"
            f"click=center  scroll=+/-0.5px  up/down=+/-0.1px  "
            f"n=next  b=back  x=bad  s=SAVE  q=quit", fontsize=10)
        fig.canvas.draw_idle()

    def on_scroll(ev):
        bd = beads[state["i"]]
        bd["r"] = max(1.0, bd["r"] + (0.5 if ev.button == "up" else -0.5))
        bd["tagged"] = True
        show()

    def on_click(ev):
        if ev.inaxes is ax and ev.xdata is not None:
            bd = beads[state["i"]]
            bd["cx"], bd["cy"] = ev.xdata, ev.ydata
            bd["tagged"] = True
            show()

    def nxt(d):
        state["i"] = (state["i"] + d) % len(beads)
        show()

    def save():
        rows = [dict(particle=b["pid"], r_px_manual=round(b["r"], 2),
                     r_um_manual=round(b["r"] * mpp, 4))
                for b in beads if b["tagged"] and not b["bad"]]
        pd.DataFrame(rows).to_csv(os.path.join(out, "radius_manual.csv"),
                                  index=False)
        print(f"[radius_tag] {stem}: wrote radius_manual.csv -- {len(rows)} beads "
              f"-> {out}\n  now: python analyze_run.py {stem}")

    def on_key(ev):
        bd = beads[state["i"]]
        if ev.key in ("n", "enter", " ", "right"):
            bd["tagged"] = True
            nxt(1)
        elif ev.key in ("b", "left"):
            nxt(-1)
        elif ev.key == "up":
            bd["r"] += 0.1; bd["tagged"] = True; show()
        elif ev.key == "down":
            bd["r"] = max(1.0, bd["r"] - 0.1); bd["tagged"] = True; show()
        elif ev.key == "x":
            bd["bad"] = not bd["bad"]; show()
        elif ev.key == "s":
            save(); plt.close(fig)
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()


if __name__ == "__main__":
    review(sys.argv[1] if len(sys.argv) > 1 else "run7")
