# -*- coding: utf-8 -*-
"""Final-report figures: (1) full-page 6-panel raw-frame gallery with a fixed
bottom-right 2.9x zoom inset per concentration; (2) tracking figure: 1x3 strip of
the 0.06% run with the enclosing circle at three times (last one = last edge-free
frame per the clipped flag) + R(t) panel with the times marked.
Sources: crops from experiments/dla-concentration-final, video + radius CSV from
experiments/week5-dla-concentration (calibration 35.6 px/mm, meta_run3.json)."""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
CROPS = r"C:/dev/brownian-motion/experiments/dla-concentration-final/figures/crops"
W5 = r"C:/dev/brownian-motion/experiments/week5-dla-concentration"
VIDEO = os.path.join(W5, "raw-videos", "run3_0.06C.mov")
CSV = os.path.join(W5, "final", "data", "radius_run3.csv")
PX_PER_MM = 35.6
CONCS = ["0.02", "0.04", "0.06", "0.15", "0.45", "0.56"]

def norm(img):
    lo, hi = np.percentile(img, [2, 99.5])
    return np.clip((img.astype(float) - lo) / (hi - lo), 0, 1)

def load_crop(c):
    img = cv2.imread(os.path.join(CROPS, "crop_c%s.png" % c), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = min(h, w)
    return norm(img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s])

def best_zoom(img, zs):
    h, w = img.shape
    best, bx, by = -1, 0, 0
    for fy in np.linspace(0.08, 0.72, 14):
        for fx in np.linspace(0.08, 0.72, 14):
            x, y = int(fx * w), int(fy * h)
            win = img[y:y + zs, x:x + zs]
            if win.shape != (zs, zs):
                continue
            if fx + zs / w > 0.52 and fy + zs / h > 0.49:   # keep clear of the fixed inset
                continue
            score = win.std() * (1 - win.mean() * 0.3)
            if score > best:
                best, bx, by = score, x, y
    return bx, by

# ---------------- Raw: full-page 2x3 gallery, zoom inset per panel ----------------
fig, axes = plt.subplots(3, 2, figsize=(8.5, 12.4))
order = ["0.02", "0.04", "0.06", "0.15", "0.45", "0.56"]
for ax, c, letter in zip(axes.ravel(), order, "abcdef"):
    img = load_crop(c)
    h, w = img.shape
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.035, 0.968, "(%s) %s%%" % (letter, c), transform=ax.transAxes,
            ha="left", va="top", fontsize=12, color="w",
            bbox=dict(facecolor="black", alpha=0.55, pad=3.5, edgecolor="none"))
    zs = int(w * 0.15)
    zx, zy = best_zoom(img, zs)
    if c == "0.02":                      # editorial pick: the richer branch below
        zy = min(zy + zs, h - zs - 1)
    spot = [0.555, 0.015, 0.43, 0.43]   # fixed: bottom-right in every panel
    bxf, byf = (zx + zs / 2) / w, 1 - (zy + zs / 2) / h   # box center, axes fraction
    axins = ax.inset_axes(spot)
    axins.imshow(img, cmap="gray", vmin=0, vmax=1)
    axins.set_xlim(zx, zx + zs)
    axins.set_ylim(zy + zs, zy)
    axins.set_xticks([]); axins.set_yticks([])
    for sp in axins.spines.values():
        sp.set_color("#e4572e"); sp.set_linewidth(1.8)
    mag = spot[2] / (zs / w)
    axins.text(0.955, 0.045, "%.1f×" % mag, transform=axins.transAxes,
               ha="right", va="bottom", fontsize=10, color="w",
               bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    ax.add_patch(Rectangle((zx, zy), zs, zs, fill=False, ec="#e4572e", lw=1.5))
    icx, icy = spot[0] + spot[2] / 2, spot[1] + spot[3] / 2
    dx, dy = bxf - icx, byf - icy
    # display-space box corners (data coords, y down): T=top of screen
    TL, TR = (zx, zy), (zx + zs, zy)
    BL, BR = (zx, zy + zs), (zx + zs, zy + zs)
    # inset corners in axes-fraction of axins ((0,0)=display bottom-left)
    iTL, iTR, iBL, iBR = (0, 1), (1, 1), (0, 0), (1, 0)
    if abs(dx) >= abs(dy):
        pairs = [(TL, iTR), (BL, iBR)] if dx > 0 else [(TR, iTL), (BR, iBL)]
    else:
        pairs = [(BL, iTL), (BR, iTR)] if dy > 0 else [(TL, iBL), (TR, iBR)]
    from matplotlib.patches import ConnectionPatch
    for (pa, pb) in pairs:
        ax.add_artist(ConnectionPatch(xyA=pa, coordsA=ax.transData,
                                      xyB=pb, coordsB=axins.transAxes,
                                      color="#e4572e", lw=1.2))
fig.subplots_adjust(wspace=0.02, hspace=0.015, left=0.005, right=0.995, top=0.998, bottom=0.002)
fig.savefig(os.path.join(OUT, "raw_frames_gallery.png"), dpi=200)
fig.savefig(os.path.join(OUT, "raw_frames_gallery.pdf"))
plt.close(fig)
print("raw full-page done")

# ---------------- Tracking C v2: strip + R(t), panel (c) at last edge-free time ----------------
d = np.genfromtxt(CSV, delimiter=",", names=True)
ok = d["Renc_px"] > 0
t, R, cx, cy = d["t_s"][ok], d["Renc_px"][ok], d["cx_px"][ok], d["cy_px"][ok]
clipped = d["clipped"][ok]
free = clipped == 0
t_free = t[free].max()
print("last edge-free t =", t_free, "of", t.max())

cap = cv2.VideoCapture(VIDEO)
FW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def grab(ts):
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
    ret, fr = cap.read()
    assert ret, ts
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

def at(ts):
    i = np.argmin(np.abs(t - ts))
    return t[i], R[i], cx[i], cy[i]

c0x = np.median(cx)
x0 = int(np.clip(c0x - FH / 2, 0, FW - FH))
CROP = (x0, 0)

def crop_frame(img):
    return norm(img[0:FH, x0:x0 + FH])

CIRC = "#e4572e"
times3 = [0.3 * t_free, 0.65 * t_free, t_free]

fig = plt.figure(figsize=(9.6, 2.95))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.42], wspace=0.10)
marks = []
for k, (ts, letter) in enumerate(zip(times3, "abc")):
    ax = fig.add_subplot(gs[0, k])
    ti, Ri, cxi, cyi = at(ts)
    marks.append((ti, Ri))
    ax.imshow(crop_frame(grab(ti)), cmap="gray", vmin=0, vmax=1)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(cxi - x0 + Ri * np.cos(th), cyi + Ri * np.sin(th), color=CIRC, lw=1.5)
    ax.set_xlim(0, FH); ax.set_ylim(FH, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.05, 0.955, "(%s) t = %d s" % (letter, round(ti)), transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="w",
            bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    if letter == "c":
        bar = 5 * PX_PER_MM
        bx, by = FH * 0.95 - bar, FH * 0.93
        ax.plot([bx, bx + bar], [by, by], color="w", lw=3, solid_capstyle="butt")
        ax.text(bx + bar / 2, by - FH * 0.02, "5 mm", color="w", ha="center",
                va="bottom", fontsize=8)
axr = fig.add_subplot(gs[0, 3])
axr.plot(t, R / PX_PER_MM, color="#33658a", lw=1.4, label="R(t)")
tclip = t[~free]
if len(tclip):
    axr.axvspan(tclip.min(), t.max(), color="#33658a", alpha=0.08)
    axr.text(tclip.min() + 12, 2.0, "edge contact", fontsize=8, color="#33658a", rotation=90)
for (ti, Ri), letter in zip(marks, "abc"):
    axr.plot(ti, Ri / PX_PER_MM, "o", color=CIRC, ms=6, zorder=5)
    axr.annotate("(%s)" % letter, (ti, Ri / PX_PER_MM), textcoords="offset points",
                 xytext=(7, -12), fontsize=9)
axr.set_xlabel("t [s]", fontsize=9)
axr.set_ylabel("R [mm]", fontsize=9, labelpad=2)
axr.yaxis.set_tick_params(pad=1)
axr.tick_params(labelsize=8)
axr.set_title("(d)", fontsize=10, loc="left")
fig.subplots_adjust(left=0.005, right=0.99, top=0.90, bottom=0.19)
fig.savefig(os.path.join(OUT, "tracking_circle_strip.png"), dpi=200)
fig.savefig(os.path.join(OUT, "tracking_circle_strip.pdf"))
plt.close(fig)
cap.release()
print("trk C v2 done")
