"""
label_beads.py  (week1-system-calibration)
------------------------------------------
One-bead-at-a-time labelling GUI for Plot-2 curation. Shows each tracked bead's
crop (contrast-stretched, no overlay); you press a key or click a button:

    perfect / singlet / doublet / blob

Writes labels.csv (particle, r_um, type) after every label, so it is fully
RESUMABLE -- rerun and it resends you to the first unlabelled bead. Beads are
shown small -> large (by measured radius). Beads near the frame edge are clamped
and flagged "[near edge]".

Keys:  1/p perfect   2/s singlet   3/d doublet   4/b blob
       <space> show another frame of this bead   <Left> back   q quit & save

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/label_beads.py run3
    python scripts/label_beads.py run3 --tag d21m600
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

import _paths

CATS = [("perfect", "1"), ("singlet", "2"), ("doublet", "3"), ("blob", "4")]


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


class Labeler:
    def __init__(self, args):
        import cv2
        self.args = args
        stem = args.run
        cdir = _paths.clip_dir(stem)
        if args.tag:
            cdir = os.path.join(cdir, args.tag)
        self.cdir = cdir
        tcsv = os.path.join(cdir, "trajectory.csv")
        if not os.path.exists(tcsv):
            sys.exit(f"no trajectory.csv in {cdir} -- run track.py first")
        self.traj = pd.read_csv(tcsv)
        counts = self.traj.groupby("particle")["frame"].count()
        pids = counts[counts >= args.min_len].index.tolist()

        rpath = os.path.join(cdir, "radius.csv")
        self.rad = pd.read_csv(rpath).set_index("particle") if os.path.exists(rpath) else None
        pids.sort(key=lambda p: (np.nan_to_num(self.rof(p), nan=1e9), p))
        self.pids = pids

        self.lpath = os.path.join(cdir, "labels.csv")
        self.labels = {}
        if os.path.exists(self.lpath):
            old = pd.read_csv(self.lpath)
            for _, r in old.iterrows():
                if isinstance(r.get("type"), str) and r["type"]:
                    self.labels[int(r["particle"])] = r["type"]

        self.i = 0
        for k, p in enumerate(pids):
            if p not in self.labels:
                self.i = k
                break

        meta = _paths.load_runs().get("runs", {}).get(stem, {}).get("video", stem + ".avi")
        self.cap = cv2.VideoCapture(_paths.video(meta))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Hgt = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cur_which = None
        self._build()
        self.show()

    def rof(self, p):
        if self.rad is not None and p in self.rad.index:
            return float(self.rad.loc[p, "r_um"])
        return np.nan

    # ---------- GUI ----------
    def _build(self):
        self.root = tk.Tk()
        self.root.title("label beads")
        self.fig = plt.Figure(figsize=(4.2, 4.2))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.status = tk.Label(self.root, text="", font=("TkDefaultFont", 11))
        self.status.pack(pady=2)
        bar = tk.Frame(self.root); bar.pack(pady=6)
        for name, key in CATS:
            tk.Button(bar, text=f"{name} ({key})", width=10,
                      command=lambda n=name: self.label(n)).pack(side=tk.LEFT, padx=3)
        nav = tk.Frame(self.root); nav.pack(pady=2)
        tk.Button(nav, text="another frame (space)", command=self.resample).pack(side=tk.LEFT, padx=3)
        tk.Button(nav, text="back (\u2190)", command=self.back).pack(side=tk.LEFT, padx=3)
        tk.Button(nav, text="quit & save (q)", command=self.quit).pack(side=tk.LEFT, padx=3)
        for name, key in CATS:
            self.root.bind(key, lambda e, n=name: self.label(n))
            self.root.bind(name[0], lambda e, n=name: self.label(n))
        self.root.bind("<space>", lambda e: self.resample())
        self.root.bind("<Left>", lambda e: self.back())
        self.root.bind("q", lambda e: self.quit())

    # ---------- image ----------
    def _frames(self, pid):
        return self.traj[self.traj.particle == pid].sort_values("frame").reset_index(drop=True)

    def crop(self, pid):
        sub = self._frames(pid)
        n = len(sub)
        idx = n // 2 if self.cur_which is None else self.cur_which % n
        row = sub.iloc[idx]
        fr = gray_frame(self.cap, int(row.frame))
        if fr is None:
            return None, False
        H = self.args.half
        x, y = float(row.x), float(row.y)
        x0, y0 = int(round(x)) - H, int(round(y)) - H
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(self.W, x0 + 2 * H + 1), min(self.Hgt, y0 + 2 * H + 1)
        crop = fr[y0c:y1c, x0c:x1c]
        clamped = x0 < 0 or y0 < 0 or x0 + 2 * H + 1 > self.W or y0 + 2 * H + 1 > self.Hgt
        lo, hi = np.percentile(crop, [2, 98])
        return np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1), clamped

    def show(self):
        if self.i >= len(self.pids):
            self.quit(); return
        pid = self.pids[self.i]
        crop, clamped = self.crop(pid)
        self.ax.clear(); self.ax.axis("off")
        if crop is not None:
            self.ax.imshow(crop, cmap="gray")
        r = self.rof(pid)
        rtxt = f"  r={r:.2f}\u00b5m" if np.isfinite(r) else ""
        edge = "  [near edge]" if clamped else ""
        cur = self.labels.get(pid, "")
        self.ax.set_title(f"p{pid}{rtxt}{edge}", fontsize=12)
        self.canvas.draw()
        done = sum(1 for p in self.pids if p in self.labels)
        self.status.config(text=f"bead {self.i+1}/{len(self.pids)}    "
                                f"labelled {done}/{len(self.pids)}"
                                + (f"    (current: {cur})" if cur else ""))

    # ---------- actions ----------
    def resample(self):
        pid = self.pids[self.i]
        n = len(self._frames(pid))
        base = n // 2 if self.cur_which is None else self.cur_which
        self.cur_which = (base + max(1, n // 6)) % n
        self.show()

    def label(self, name):
        self.labels[self.pids[self.i]] = name
        self.save()
        self.i += 1
        self.cur_which = None
        self.show()

    def back(self):
        self.i = max(0, self.i - 1)
        self.cur_which = None
        self.show()

    def save(self):
        rows = [{"particle": p, "r_um": self.rof(p), "type": self.labels[p]}
                for p in self.pids if p in self.labels]
        pd.DataFrame(rows).to_csv(self.lpath, index=False)

    def quit(self, *a):
        self.save()
        try:
            self.cap.release()
        except Exception:
            pass
        print(f"[label] saved {self.lpath} ({len(self.labels)}/{len(self.pids)} labelled)")
        self.root.quit()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="One-at-a-time bead labelling GUI.")
    ap.add_argument("run", help="run stem, e.g. run3")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--half", type=int, default=22, help="crop half-width (px)")
    args = ap.parse_args()
    Labeler(args).root.mainloop()


if __name__ == "__main__":
    main()
