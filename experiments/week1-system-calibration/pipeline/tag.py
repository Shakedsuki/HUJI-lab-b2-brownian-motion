"""
tag.py  (pipeline)
------------------
Interactive review/tagging GUI for singleton curation. Streamlines the human
confirm step: one bead at a time, single-key verdicts, with a free-text NOTE for
cases that don't fit the canonical buckets. Autosaves to labels.csv (resumable).

Per bead it shows the START / MIDDLE / END frame crops (so defocus or a doublet
that only separates late is visible) + the fitted ring + the auto metrics.

Keys
----
  1 / k     keep  -> clean single
  0 / x     reject (generic)
  d         reject: doublet        b   reject: blob
  f         reject: defocus        o   reject: other
  n         NOTE: type free text, Enter to save, Esc to cancel
  →/space   next        ←   previous        u  toggle keep
  s         save now    q   save & quit      ?  toggle help

Run (from the week root)
------------------------
    python -m pipeline.tag run3                       # tags measurements/run3/pipeline/
    python -m pipeline.tag run3 --dir measurements/run3/pipeline/early_tag
    python -m pipeline.tag run3 --selftest            # headless check, no window

Writes <dir>/labels.csv with columns: particle, keep, type, note, + the metrics
+ auto proposal. keep==1 is what the rest of the pipeline consumes.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from . import paths, shape

FEAT = ["x_med", "y_med", "R_px_med", "sym_med", "ecc_med", "ring_cv_med",
        "resid_med", "n_cores_med", "proposed", "reason"]
HELP = ("[1/k]keep  [0/x]reject  [d]oublet [b]lob [f]defocus [o]ther   "
        "[n]ote  [←→/space]nav  [u]toggle  [s]ave  [q]uit  [?]help")


def _read_crops(vid, flat, reqs, half=24):
    """reqs: list of (key, frame, x, y) -> {key: (crop, cx, cy)}; each frame read once."""
    import cv2
    by_frame = {}
    for key, f, x, y in reqs:
        by_frame.setdefault(int(f), []).append((key, x, y))
    crops = {}
    cap = cv2.VideoCapture(vid)
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


class Tagger:
    REJECT_KEYS = {"d": "doublet", "b": "blob", "f": "defocus", "o": "other"}

    def __init__(self, stem, tag_dir, videos_dir=None):
        self.stem = stem
        self.dir = tag_dir
        cpath = os.path.join(tag_dir, "curation.csv")
        if not os.path.exists(cpath):
            raise SystemExit(f"[tag] no curation.csv in {tag_dir}")
        self.cur = pd.read_csv(cpath)
        self.traj = pd.read_csv(os.path.join(tag_dir, "trajectory.csv")) \
            if os.path.exists(os.path.join(tag_dir, "trajectory.csv")) \
            else pd.read_csv(os.path.join(paths.out_dir(stem), "trajectory.csv"))
        self.vid = paths.video(paths.video_for_run(stem), videos_dir)
        self.pids = self.cur["particle"].astype(int).tolist()
        self.idx = 0
        self.note_mode = False
        self.note_buf = ""
        self._init_decisions()
        self._build_crops()

    def _init_decisions(self):
        self.dec = {}
        for _, r in self.cur.iterrows():
            pid = int(r["particle"])
            single = (r["proposed"] == "single")
            self.dec[pid] = dict(keep=int(single),
                                 type=("single" if single else "reject"),
                                 note="")
        lp = os.path.join(self.dir, "labels.csv")
        if os.path.exists(lp):                              # resume prior tags
            old = pd.read_csv(lp)
            for _, r in old.iterrows():
                pid = int(r["particle"])
                if pid in self.dec:
                    self.dec[pid] = dict(keep=int(r.get("keep", 0)),
                                         type=str(r.get("type", "")),
                                         note=("" if pd.isna(r.get("note", ""))
                                               else str(r.get("note", ""))))
            print(f"[tag] resumed {len(old)} prior labels from {lp}")

    def _build_crops(self):
        from . import frames as fr
        print("[tag] building flat-field + crops (one-time)...")
        flat = fr.flat_field(self.vid, n_sample=60)
        reqs, self.frames_of = [], {}
        for pid in self.pids:
            g = self.traj[self.traj["particle"] == pid].sort_values("frame")
            if len(g) == 0:
                self.frames_of[pid] = []
                continue
            picks = g.iloc[np.linspace(0, len(g) - 1, 3).astype(int)]
            keys = []
            for _, row in picks.iterrows():
                k = (pid, int(row["frame"]))
                reqs.append((k, int(row["frame"]), float(row["x_raw"]),
                             float(row["y_raw"])))
                keys.append(k)
            self.frames_of[pid] = keys
            # representative shape (fit circle on the middle pick)
            mid = picks.iloc[len(picks) // 2]
            self.cur.loc[self.cur["particle"] == pid, "_polz"] = int(mid["polarity"])
        self.crops = _read_crops(self.vid, flat, reqs)
        print(f"[tag] cached crops for {len(self.pids)} beads")

    # ---- persistence -----------------------------------------------------
    def save(self):
        rows = []
        for _, r in self.cur.iterrows():
            pid = int(r["particle"])
            d = self.dec[pid]
            rows.append(dict(particle=pid, keep=d["keep"], type=d["type"],
                             note=d["note"],
                             **{c: r[c] for c in FEAT if c in r}))
        pd.DataFrame(rows).to_csv(os.path.join(self.dir, "labels.csv"), index=False)

    # ---- GUI -------------------------------------------------------------
    def run(self):
        import matplotlib
        ok = False
        for bk in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
            try:
                matplotlib.use(bk, force=True)
                ok = True
                break
            except Exception:                               # noqa: BLE001
                continue
        import matplotlib.pyplot as plt
        bk_now = matplotlib.get_backend()
        # NB: only the bare "Agg" backend is non-interactive; "TkAgg"/"QtAgg"
        # also contain "agg" but ARE interactive -> exact match, not substring.
        if not ok or bk_now.lower() == "agg":
            sys.exit(f"[tag] no interactive matplotlib backend (got {bk_now}). "
                     f"Install tkinter/PyQt or set MPLBACKEND.")
        print(f"[tag] backend: {bk_now}")
        self.plt = plt
        self.fig, self.axes = plt.subplots(1, 3, figsize=(11, 4.6))
        self.fig.subplots_adjust(top=0.80, bottom=0.16)
        self.ims, self.circs = [], []
        th = np.linspace(0, 2 * np.pi, 90)
        self._th = th
        for ax in self.axes:
            ax.set_xticks([]); ax.set_yticks([])
            self.ims.append(ax.imshow(np.zeros((10, 10)), cmap="gray"))
            (c,) = ax.plot([], [], "-", color="orange", lw=1.2)
            self.circs.append(c)
        self.txt = self.fig.text(0.5, 0.965, "", ha="center", va="top",
                                 fontsize=10)
        self.foot = self.fig.text(0.5, 0.04, HELP, ha="center", va="bottom",
                                  fontsize=7.5, color="0.3")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()
        plt.show()
        self.save()
        print(f"[tag] saved labels.csv ({sum(d['keep'] for d in self.dec.values())} "
              f"keep) -> {self.dir}")

    def _render(self):
        pid = self.pids[self.idx]
        d = self.dec[pid]
        keys = self.frames_of.get(pid, [])
        labels = ["start", "mid", "end"]
        for j, ax in enumerate(self.axes):
            if j < len(keys) and keys[j] in self.crops:
                crop, cx, cy = self.crops[keys[j]]
                self.ims[j].set_data(crop)
                self.ims[j].set_clim(crop.min(), crop.max())
                self.ims[j].set_extent([0, crop.shape[1], crop.shape[0], 0])
                ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
                ax.set_title(labels[j], fontsize=8)
            else:
                self.ims[j].set_data(np.zeros((10, 10)))
            self.circs[j].set_data([], [])
        # fit circle on the middle panel
        if len(keys) >= 2 and keys[len(keys) // 2] in self.crops:
            crop, cx, cy = self.crops[keys[len(keys) // 2]]
            row = self.cur[self.cur["particle"] == pid].iloc[0]
            pol = int(row.get("_polz", 1))
            m = shape.measure_shape(crop, cx, cy, max(row["R_px_med"], 4), pol)
            if np.isfinite(m["R"]):
                cc = m["_cxy"]
                self.circs[len(keys) // 2].set_data(
                    cc[0] + m["R"] * np.cos(self._th),
                    cc[1] + m["R"] * np.sin(self._th))
        r = self.cur[self.cur["particle"] == pid].iloc[0]
        verdict = ("KEEP single" if d["keep"] else f"REJECT [{d['type']}]")
        color = "green" if d["keep"] else "red"
        head = (f"[{self.idx+1}/{len(self.pids)}] p{pid}   "
                f"auto={r['proposed']} ({str(r['reason'])[:24] or 'ok'})   "
                f"r={r['R_px_med']*0.14381:.2f}um ecc={r['ecc_med']:.2f} "
                f"ringcv={r['ring_cv_med']:.2f} resid={r['resid_med']:.2f} "
                f"cores={r['n_cores_med']:.0f}")
        if self.note_mode:
            sub = f"NOTE (Enter=save, Esc=cancel): {self.note_buf}_"
        else:
            sub = f">>> {verdict}" + (f"   note: {d['note']}" if d["note"] else "")
        self.txt.set_text(head + "\n" + sub)
        self.txt.set_color("black" if self.note_mode else color)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        pid = self.pids[self.idx]
        k = event.key
        if self.note_mode:
            if k == "enter":
                self.dec[pid]["note"] = self.note_buf
                self.note_mode = False; self.save()
            elif k == "escape":
                self.note_mode = False
            elif k == "backspace":
                self.note_buf = self.note_buf[:-1]
            elif k == "space":
                self.note_buf += " "
            elif k and len(k) == 1:
                self.note_buf += k
            self._render(); return

        if k in ("right", " ", "space"):
            self.idx = min(self.idx + 1, len(self.pids) - 1)
        elif k == "left":
            self.idx = max(self.idx - 1, 0)
        elif k in ("1", "k"):
            self.dec[pid].update(keep=1, type="single"); self.save()
            self.idx = min(self.idx + 1, len(self.pids) - 1)
        elif k in ("0", "x"):
            self.dec[pid].update(keep=0,
                                 type=("reject" if self.dec[pid]["type"] == "single"
                                       else self.dec[pid]["type"])); self.save()
            self.idx = min(self.idx + 1, len(self.pids) - 1)
        elif k in self.REJECT_KEYS:
            self.dec[pid].update(keep=0, type=self.REJECT_KEYS[k]); self.save()
            self.idx = min(self.idx + 1, len(self.pids) - 1)
        elif k == "u":
            self.dec[pid]["keep"] ^= 1
            self.dec[pid]["type"] = "single" if self.dec[pid]["keep"] else "reject"
            self.save()
        elif k == "n":
            self.note_mode = True
            self.note_buf = self.dec[pid]["note"]
        elif k == "s":
            self.save(); print("[tag] saved")
        elif k == "q":
            self.plt.close(self.fig); return
        elif k == "?":
            self.foot.set_visible(not self.foot.get_visible())
        self._render()


def selftest(stem, tag_dir):
    """Headless check of load/crop/render/save (no window)."""
    import matplotlib
    matplotlib.use("Agg")
    t = Tagger.__new__(Tagger)
    t.stem, t.dir = stem, tag_dir
    t.cur = pd.read_csv(os.path.join(tag_dir, "curation.csv")).head(6)
    t.traj = pd.read_csv(os.path.join(paths.out_dir(stem), "trajectory.csv"))
    t.vid = paths.video(paths.video_for_run(stem))
    t.pids = t.cur["particle"].astype(int).tolist()
    t.idx = 0; t.note_mode = False; t.note_buf = ""
    t._init_decisions(); t._build_crops()
    import matplotlib.pyplot as plt
    t.plt = plt
    t.fig, t.axes = plt.subplots(1, 3)
    t.ims = [ax.imshow(np.zeros((10, 10))) for ax in t.axes]
    t.circs = [ax.plot([], [])[0] for ax in t.axes]
    t._th = np.linspace(0, 2 * np.pi, 90)
    t.txt = t.fig.text(0.5, 0.96, ""); t.foot = t.fig.text(0.5, 0.04, "")
    t._render()
    t.dec[t.pids[0]].update(keep=1, type="single", note="selftest note, comma, ok")
    t.save()
    out = pd.read_csv(os.path.join(tag_dir, "labels.csv"))
    print(f"[selftest] OK: cached {len(t.crops)} crops, rendered bead 0, "
          f"wrote labels.csv ({len(out)} rows, cols={list(out.columns)})")
    plt.close(t.fig)


def main():
    ap = argparse.ArgumentParser(description="Interactive bead tagging GUI.")
    ap.add_argument("run")
    ap.add_argument("--dir", default=None, help="folder with curation.csv "
                    "(default measurements/<run>/pipeline)")
    ap.add_argument("--videos-dir", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    tag_dir = args.dir or paths.out_dir(args.run)
    if args.selftest:
        selftest(args.run, tag_dir)
    else:
        Tagger(args.run, tag_dir, videos_dir=args.videos_dir).run()


if __name__ == "__main__":
    main()
