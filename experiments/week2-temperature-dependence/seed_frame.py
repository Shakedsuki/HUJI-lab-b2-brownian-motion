"""
seed_frame.py <run> [--frame N]
-------------------------------
Reference-frame grounding GUI. Shows the SHARPEST full frame with auto-detected
candidates drawn as faint circles. You CLICK the good single spheres to keep them
(green), resize each to the TRUE sphere edge, and add any missed singles; only
kept circles are saved. Those seeds (x, y, r) then select + size the tracked beads
(seed_match.py), so the human grounds purity AND radius up front -- on a frame,
blind to motion, so it can't bias D.

  left-drag ........ zoom (matplotlib toolbar; the home icon resets view)
  right-click ...... on a candidate = KEEP + select (green); on empty = add a seed
  scroll ........... resize selected   +/- 0.5 px
  up / down ........ resize selected   +/- 0.1 px (fine)
  d ................ drop selected (back to a faint candidate)
  s ................ SAVE -> seeds.csv  (then closes)
  q ................ quit without saving

Then:  python track_week2.py <run> --frames 1000 --force   (if not tracked yet)
       python seed_match.py <run>
       python analyze_run.py <run>
"""
import os
import sys

import numpy as np
import pandas as pd

from pipeline import paths, frames as fr, detect


def sharpest_frame(vid, n=40, lo=0, hi=None):
    """Frame with the highest global Laplacian variance over n samples in
    [lo, hi] -- constrained to the tracked range so seed_match can find it."""
    import cv2
    cap = cv2.VideoCapture(vid)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hi = (tot - 1) if hi is None else min(hi, tot - 1)
    idxs = np.unique(np.linspace(lo, max(lo, hi), n).astype(int))
    best_v, best_i = -1.0, 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frm = cap.read()
        if not ok:
            continue
        g = frm[..., :3].mean(-1).astype(np.float32)
        v = float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())
        if v > best_v:
            best_v, best_i = v, int(i)
    cap.release()
    return best_i


def review(stem, frame=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    out = paths.out_dir(stem)
    mpp = paths.load_scale() or 0.14381
    vid = paths.video(paths.video_for_run(stem))
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"))
    if frame is None:
        # constrain the reference frame to what is (or will be) tracked, so
        # seed_match can locate the seeded beads in trajectory.csv
        tcsv = os.path.join(out, "trajectory.csv")
        if os.path.exists(tcsv):
            fcol = pd.read_csv(tcsv, usecols=["frame"])["frame"]
            lo_f, hi_f = int(fcol.min()), int(fcol.max())
        else:
            lo_f, hi_f = 0, min(fr.count_frames(vid) - 1, 999)  # default track window
        print(f"[seed_frame] finding sharpest frame in [{lo_f},{hi_f}]...", flush=True)
        frame = sharpest_frame(vid, lo=lo_f, hi=hi_f)
    print(f"[seed_frame] reference frame {frame}; detecting candidates...", flush=True)
    raw = fr.frame_at(vid, frame)
    d = detect.detect_frame(raw - flat, sym_min=0.18, grad_pct=80.0)
    seeds = [dict(x=float(d["x"][i]), y=float(d["y"][i]),
                  r=float(max(d["r_est"][i], 4.0)), kept=False)
             for i in range(len(d["x"]))]
    print(f"[seed_frame] {len(seeds)} candidates -- click the good single spheres "
          f"to keep (green) + resize, then 's'", flush=True)

    sel = {"i": None}
    lo, hi = np.percentile(raw, (1, 99))
    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.imshow(raw, cmap="gray", vmin=lo, vmax=hi)
    ax.set_xticks([]); ax.set_yticks([])
    circles = []

    def title():
        nk = sum(s["kept"] for s in seeds)
        nsel = "" if sel["i"] is None else \
            f"   sel r={seeds[sel['i']]['r']:.1f}px={seeds[sel['i']]['r'] * mpp:.3f}um"
        ax.set_title(f"{stem} frame {frame}: {nk} kept / {len(seeds)} candidates"
                     f"{nsel}\nright-click a sphere=keep+select  scroll/up-down="
                     f"radius  d=drop  s=SAVE  q=quit", fontsize=10)

    def add_circle(i):
        s = seeds[i]
        c = Circle((s["x"], s["y"]), s["r"], fill=False, edgecolor="#888888",
                   lw=0.6, alpha=0.5)
        ax.add_patch(c)
        circles.append(c)

    def restyle():
        for i, c in enumerate(circles):
            on = (i == sel["i"])
            if seeds[i]["kept"]:
                c.set_edgecolor("#33dd33")
                c.set_linewidth(2.6 if on else 1.4)
                c.set_alpha(1.0)
            else:
                c.set_edgecolor("#ffd000" if on else "#888888")
                c.set_linewidth(2.0 if on else 0.6)
                c.set_alpha(0.9 if on else 0.45)
            c.set_radius(seeds[i]["r"])
        title()
        fig.canvas.draw_idle()

    for i in range(len(seeds)):
        add_circle(i)

    def on_click(ev):
        if ev.button != 3 or ev.inaxes is not ax or ev.xdata is None:
            return
        if seeds:
            dd = [np.hypot(s["x"] - ev.xdata, s["y"] - ev.ydata) for s in seeds]
            j = int(np.argmin(dd))
        else:
            dd, j = [1e9], None
        if j is not None and dd[j] <= seeds[j]["r"] + 4:
            seeds[j]["kept"] = True            # promote candidate to a kept seed
            sel["i"] = j
        else:
            seeds.append(dict(x=float(ev.xdata), y=float(ev.ydata), r=8.0, kept=True))
            add_circle(len(seeds) - 1)
            sel["i"] = len(seeds) - 1
        restyle()

    def on_scroll(ev):
        if sel["i"] is None:
            return
        seeds[sel["i"]]["r"] = max(2.0, seeds[sel["i"]]["r"]
                                   + (0.5 if ev.button == "up" else -0.5))
        restyle()

    def on_key(ev):
        i = sel["i"]
        if ev.key == "up" and i is not None:
            seeds[i]["r"] += 0.1; restyle()
        elif ev.key == "down" and i is not None:
            seeds[i]["r"] = max(2.0, seeds[i]["r"] - 0.1); restyle()
        elif ev.key == "d" and i is not None:
            seeds[i]["kept"] = False            # drop back to a faint candidate
            sel["i"] = None; restyle()
        elif ev.key == "s":
            save(); plt.close(fig)
        elif ev.key == "q":
            plt.close(fig)

    def save():
        rows = [dict(frame=int(frame), x=round(s["x"], 2), y=round(s["y"], 2),
                     r_px=round(s["r"], 2), r_um=round(s["r"] * mpp, 4))
                for s in seeds if s["kept"]]
        pd.DataFrame(rows).to_csv(os.path.join(out, "seeds.csv"), index=False)
        print(f"[seed_frame] {stem}: wrote seeds.csv -- {len(rows)} kept seeds @ "
              f"frame {frame} -> {out}")
        print(f"  next: python track_week2.py {stem} --frames 1000 --force"
              f"  (skip if already tracked)")
        print(f"        python seed_match.py {stem}")
        print(f"        python analyze_run.py {stem}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    restyle()
    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Reference-frame seed/radius GUI.")
    ap.add_argument("run", nargs="?", default="run7")
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args()
    review(args.run, args.frame)
