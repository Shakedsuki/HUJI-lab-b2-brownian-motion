# -*- coding: utf-8 -*-
"""Merged figure: tracking frames (a-c, 0.06% run, enclosing circle, last frame =
last edge-free time) above the six-panel shared-axes R(t) grid (d-i). The 0.06%
panel is outlined in orange and carries labeled markers at the frame times.
Replaces the separate tracking and R(t)-grid figures in the final report."""
import os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
sys.path.insert(0, r"C:/dev/brownian-motion/experiments/dla-concentration-final/scripts")
import report_figures as rf

W5 = r"C:/dev/brownian-motion/experiments/week5-dla-concentration"
VIDEO = os.path.join(W5, "raw-videos", "run3_0.06C.mov")
CSV = os.path.join(W5, "final", "data", "radius_run3.csv")
PX_PER_MM = 35.6
CIRC = "#e4572e"
BLUE = "#33658a"

runs = [rf.load_run(r) for r in rf.RUNS]

def norm(img):
    lo, hi = np.percentile(img, [2, 99.5])
    return np.clip((img.astype(float) - lo) / (hi - lo), 0, 1)

# ---- tracking-run data + frames (same as approved tracking figure)
d = np.genfromtxt(CSV, delimiter=",", names=True)
ok = d["Renc_px"] > 0
t, R, cx, cy = d["t_s"][ok], d["Renc_px"][ok], d["cx_px"][ok], d["cy_px"][ok]
clipped = d["clipped"][ok]
t_free = t[clipped == 0].max()
cap = cv2.VideoCapture(VIDEO)
FH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
c0x = np.median(cx)
x0 = int(np.clip(c0x - FH / 2, 0, FW - FH))

def grab(ts):
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
    ret, fr = cap.read()
    assert ret
    return norm(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[0:FH, x0:x0 + FH])

def at(ts):
    i = np.argmin(np.abs(t - ts))
    return t[i], R[i], cx[i], cy[i]

times3 = [0.3 * t_free, 0.65 * t_free, t_free]

def draw_frame(ax, ts, letter):
    ti, Ri, cxi, cyi = at(ts)
    ax.imshow(grab(ti), cmap="gray", vmin=0, vmax=1)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(cxi - x0 + Ri * np.cos(th), cyi + Ri * np.sin(th), color=CIRC, lw=1.5)
    ax.set_xlim(0, FH); ax.set_ylim(FH, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.05, 0.955, "(%s) t = %d s" % (letter, round(ti)), transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="w",
            bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    return ti, Ri

# =========== Direction A: frames on top, full shared-axes R(t) grid below ===========
fig = plt.figure(figsize=(9, 8.6))
gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1, 1], hspace=0.22, wspace=0.18,
                      left=0.065, right=0.985, top=0.985, bottom=0.06)
marks = []
for k, (ts, letter) in enumerate(zip(times3, "abc")):
    ax = fig.add_subplot(gs[0, k])
    marks.append(draw_frame(ax, ts, letter))
    if letter == "c":
        bar = 5 * PX_PER_MM
        bx, by = FH * 0.95 - bar, FH * 0.93
        ax.plot([bx, bx + bar], [by, by], color="w", lw=3, solid_capstyle="butt")
        ax.text(bx + bar / 2, by - FH * 0.02, "5 mm", color="w", ha="center", va="bottom", fontsize=8)
tmax_all = max(r["t"].max() for r in runs)
Rmax_all = max((r["Rc"] / r["ppm"]).max() for r in runs)
for k, r in enumerate(runs):
    ax = fig.add_subplot(gs[1 + k // 3, k % 3])
    tt, RR, edge = r["t"], r["Rc"] / r["ppm"], r["edge"].astype(bool)
    ax.plot(tt[~edge], RR[~edge], color=BLUE, lw=1.2)
    if edge.any():
        ax.plot(tt[edge], RR[edge], color=BLUE, lw=1.2, ls="--")
    ax.set_xlim(0, tmax_all * 1.02); ax.set_ylim(0, Rmax_all * 1.06)
    lab = "(%s) %.2f%%" % ("defghi"[k], r["conc"])
    ax.text(0.04, 0.94, lab, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    if abs(r["conc"] - 0.06) < 1e-9:
        for (ti, Ri), letter in zip(marks, "abc"):
            Rmm = Ri / PX_PER_MM
            ax.plot(ti, Rmm, "o", color=CIRC, ms=5, zorder=5)
            ax.annotate("(%s)" % letter, (ti, Rmm), textcoords="offset points",
                        xytext=(5, -11), fontsize=8, zorder=5)
        for sp in ax.spines.values():
            sp.set_color(CIRC); sp.set_linewidth(1.6)
    if k % 3 == 0:
        ax.set_ylabel("R [mm]", fontsize=9)
    if k // 3 == 1:
        ax.set_xlabel("t [s]", fontsize=9)
    ax.tick_params(labelsize=8)
fig.savefig(os.path.join(OUT, "R_vs_t_with_tracking.png"), dpi=200)
fig.savefig(os.path.join(OUT, "R_vs_t_with_tracking.pdf"))
plt.close(fig)
print("A done")

cap.release()
print('merged figure done')
