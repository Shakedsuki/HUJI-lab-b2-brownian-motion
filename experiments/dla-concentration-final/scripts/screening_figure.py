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
    # common crop box = deposit bounding box + 8 % pad
    ys, xs = np.nonzero(mask)
    pad = int(0.08 * max(np.ptp(ys), np.ptp(xs)))
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    m = mask[y0:y1, x0:x1]
    rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    return dict(mask=m, rgb=rgb, ppm=ppm, **p)


def render(data, kind):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.9))
    for ax, d in zip(axes, data):
        img = d["mask"] if kind == "mask" else d["rgb"]
        H, W = img.shape[:2]
        wmm, hmm = W / d["ppm"], H / d["ppm"]
        img = np.flipud(img)                          # origin lower -> y increases up
        if kind == "mask":
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, origin="lower",
                      extent=[0, wmm, 0, hmm], interpolation="nearest")
        else:
            ax.imshow(img, origin="lower", extent=[0, wmm, 0, hmm])
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
