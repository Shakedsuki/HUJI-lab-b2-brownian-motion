#!/usr/bin/env python3
"""Open-vs-dense deposit comparison (why D > DLA 1.71) — two clean panels.

Bare scientific two-panel figure (no on-figure prose; caption goes in the
report): our real 0.02 % deposit (open, D=1.69 ~ DLA) vs 0.06 % (dense, D=1.87).
Two versions are written:
  * screening_masks.png/pdf  -- black-on-white segmented deposit
  * screening_crops.png/pdf  -- colour video crop
Each panel has mm axes and a minimal "conc + D" tag. Needs WEEK5_VIDEO_DIR + ffmpeg.
"""

import csv, importlib.util, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W5 = ROOT.parent / "week5-dla-concentration" / "version 2"
FIGS = ROOT / "figures"
VDIR5 = Path(os.environ.get("WEEK5_VIDEO_DIR",
             r"C:\dev\brownian-motion\experiments\week5-dla-concentration\raw-videos"))

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "axes.linewidth": 1.0,
})


def load_er(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
ER = load_er(W5 / "scripts" / "enclosing_radius.py", "er_w5")

PANELS = [
    dict(letter="a", conc=0.02, vid="run1_0.02con.mov", tag="run1_c0.02", D=1.69),
    dict(letter="b", conc=0.06, vid="run3_0.06C.mov", tag="run3_c0.06", D=1.87),
]


def occ(er, img):
    fn = getattr(er, "occluder_mask", None) or getattr(er, "wire_mask")
    return fn(img)


def faithful_mask(er, bgr, ref):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    score = 1.0 - gray / (er._flatfield_bg(gray) + 1e-6)
    darkened = ref - (gray - (np.median(gray) - np.median(ref)))
    m = (er._hysteresis(score, er.HYST_HI, er.HYST_LO).astype(np.uint8)
         & (darkened > er.CHANGE_THR)).astype(np.uint8)
    o = (occ(er, bgr) & (darkened <= er.HOLE_DARK)).astype(np.uint8)
    m[o > 0] = 0; m[er.blue_grid_mask(bgr) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        sel = lab == i
        if st[i, cv2.CC_STAT_AREA] >= er.MIN_SIZE and score[sel].max() >= er.STRONG_CORE:
            out[sel] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(out)
    if n > 1:
        out = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return out


def grab(p):
    ppm = None
    for line in open(W5 / "data" / f"radius_{p['tag']}.csv"):
        if line.startswith("# px_per_mm"):
            ppm = float(line.split("=")[1].split("+/-")[0])
    r = list(csv.DictReader(l for l in open(W5 / "data" / f"radius_{p['tag']}.csv")
                            if not l.startswith("#")))
    t = max(float(x["t_s"]) for x in r if int(float(x["edge"])) == 0 and float(x["M_px"]) > 0)
    tmp = Path(tempfile.mkdtemp(prefix="scr_")); rd = tmp / "ref"; rd.mkdir()
    path = VDIR5 / p["vid"]
    subprocess.run([ER.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-frames:v", str(ER.REF_N), str(rd / "r_%03d.png")], check=True)
    ref = np.median(np.stack([cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                              for q in sorted(rd.glob("r_*.png"))]), axis=0)
    fp = tmp / "f.png"
    subprocess.run([ER.FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
                    "-i", str(path), "-frames:v", "1", str(fp)], check=True)
    bgr = cv2.imread(str(fp))
    mask = faithful_mask(ER, bgr, ref)
    shutil.rmtree(tmp, ignore_errors=True)
    ys, xs = np.nonzero(mask)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    ext = int(max(np.ptp(ys), np.ptp(xs)))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return dict(full_mask=mask, full_rgb=rgb, cx=cx, cy=cy, ext=ext, ppm=ppm, **p)


def crop_square(arr, cx, cy, side, fill):
    """A `side`x`side` window centred on (cx, cy), cut from the REAL frame so the
    surround is genuine paper (no synthetic seam). Out-of-frame shortfall (rare)
    is filled with `fill`."""
    H, W = arr.shape[:2]
    half = side // 2
    y0, x0 = cy - half, cx - half
    # keep the window inside the frame where it fits -> real paper, no fill strip
    if side <= H:
        y0 = min(max(y0, 0), H - side)
    if side <= W:
        x0 = min(max(x0, 0), W - side)
    if arr.ndim == 2:
        out = np.full((side, side), fill, arr.dtype)
    else:
        out = np.empty((side, side, 3), arr.dtype); out[:] = fill
    vy0, vy1 = max(0, y0), min(H, y0 + side)
    vx0, vx1 = max(0, x0), min(W, x0 + side)
    out[vy0 - y0:vy1 - y0, vx0 - x0:vx1 - x0] = arr[vy0:vy1, vx0:vx1]
    return out


def render(data, kind):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.9))
    # one common square window (px) + common calibration -> identical panel boxes,
    # each deposit at its true relative size (directly comparable).
    frame_min = min(min(d["full_mask"].shape) for d in data)
    side = min(int(max(d["ext"] for d in data) * 1.15), frame_min)
    ppm = float(np.mean([d["ppm"] for d in data]))
    mm = side / ppm
    for ax, d in zip(axes, data):
        if kind == "mask":
            img = crop_square(d["full_mask"], d["cx"], d["cy"], side, 0)
        else:
            fill = np.median(d["full_rgb"][:20], axis=(0, 1)).astype(d["full_rgb"].dtype)
            img = crop_square(d["full_rgb"], d["cx"], d["cy"], side, fill)
        img = np.flipud(img)                          # origin lower -> y increases up
        if kind == "mask":
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, origin="lower",
                      extent=[0, mm, 0, mm], interpolation="nearest")
        else:
            ax.imshow(img, origin="lower", extent=[0, mm, 0, mm])
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        tagcol = "black" if kind == "mask" else "white"
        ax.text(0.035, 0.965, f"({d['letter']})  {d['conc']:.2f} %   D = {d['D']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=13,
                fontweight="bold", color=tagcol,
                bbox=dict(boxstyle="round,pad=0.3", fc="black" if kind != "mask" else "white",
                          ec="0.5", alpha=0.55 if kind != "mask" else 0.85, lw=0.8))
    fig.tight_layout()
    stem = f"screening_{'masks' if kind == 'mask' else 'crops'}"
    fig.savefig(FIGS / f"{stem}.png"); fig.savefig(FIGS / f"{stem}.pdf")
    plt.close(fig)
    print(f"-> {FIGS / stem}.png/.pdf")


def main():
    data = [grab(p) for p in PANELS]
    render(data, "mask")
    render(data, "crop")
    # remove the old verbose figure
    for ext in (".png", ".pdf"):
        old = FIGS / f"screening_DLA_vs_dense{ext}"
        if old.exists():
            old.unlink()


if __name__ == "__main__":
    main()
