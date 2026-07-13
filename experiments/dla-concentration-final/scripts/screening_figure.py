#!/usr/bin/env python3
"""Educational figure: idealized DLA vs our dense deposits — why D > 1.71.

Grounded on REAL segmented deposits (not a simulation): the open 0.02 % run
(D=1.69, near DLA) vs the dense 0.06 % run (D=1.87). Overlays the screening
mechanism (strong vs weak) and anchors on our measured occupancy phi and D.
Needs WEEK5_VIDEO_DIR + ffmpeg.
"""

import csv, importlib.util, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W5 = ROOT.parent / "week5-dla-concentration" / "version 2"
FIGS = ROOT / "figures"
VDIR5 = Path(os.environ.get("WEEK5_VIDEO_DIR",
             r"C:\dev\brownian-motion\experiments\week5-dla-concentration\raw-videos"))


def load_er(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
ER = load_er(W5 / "scripts" / "enclosing_radius.py", "er_w5")

# real measured values (reliable bucket + fill-fraction)
PANELS = [
    dict(conc=0.02, vid="run1_0.02con.mov", tag="run1_c0.02", D=1.69, phi=0.16,
         kind="DIFFUSION-LIMITED  (ideal DLA)", accent="#1f77b4",
         note="strong screening", detail="tips catch ions first — interior stays empty",
         interior="open interior", branch="sparse, open branches", ref="≈ DLA 1.71"),
    dict(conc=0.06, vid="run3_0.06C.mov", tag="run3_c0.06", D=1.87, phi=0.61,
         kind="MIGRATION / CONVECTION-DRIVEN  (higher conc.)", accent="#d62728",
         note="weak screening", detail="ions add throughout — interior fills in",
         interior="filled interior", branch="dense, space-filling branches",
         ref="→ compact 2"),
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


def grab_mask(p):
    t = ppm = None
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
    mask = faithful_mask(ER, cv2.imread(str(fp)), ref)
    shutil.rmtree(tmp, ignore_errors=True)
    ys, xs = np.nonzero(mask)
    pad = int(0.10 * max(np.ptp(ys), np.ptp(xs)))
    y0, y1 = max(0, ys.min() - pad), ys.max() + pad
    x0, x1 = max(0, xs.min() - pad), xs.max() + pad
    return mask[y0:y1, x0:x1]


def draw_panel(ax, mask, p):
    H, W = mask.shape
    ax.imshow(mask, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xlim(-0.28 * W, 1.28 * W); ax.set_ylim(1.28 * H, -0.28 * H)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(p["accent"]); s.set_linewidth(2.5)
    cx, cy = W / 2, H / 2
    R = 0.5 * max(H, W)
    # incoming-ion arrows (schematic): 10 arrows aimed at the deposit
    for k, th in enumerate(np.linspace(0, 2 * np.pi, 10, endpoint=False)):
        r0, r1 = 1.24 * R, 1.03 * R
        x0, y0 = cx + r0 * np.cos(th), cy + r0 * np.sin(th)
        x1, y1 = cx + r1 * np.cos(th), cy + r1 * np.sin(th)
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=11, lw=1.3, color=p["accent"], alpha=0.6))
    # title + run label
    ax.set_title(p["kind"], fontsize=13.5, color=p["accent"], fontweight="bold", pad=12)
    ax.text(0.03, 0.98, f"our {p['conc']:.2f}% run", transform=ax.transAxes,
            ha="left", va="top", fontsize=11.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=1))
    ax.text(0.5, 0.02, "← incoming Cu²⁺ ions (random walk) →", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9.5, color=p["accent"], style="italic")
    # feature callouts to the real deposit
    ax.annotate(p["branch"], xy=(cx + 0.35 * R, cy - 0.35 * R),
                xytext=(1.05 * W, 0.12 * H), fontsize=10.5, color="0.15",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.3))
    ax.annotate(p["interior"], xy=(cx, cy), xytext=(-0.27 * W, 0.95 * H),
                fontsize=10.5, color="0.15",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.3))
    # consolidated banner below the panel: screening -> mechanism -> our values
    banner = (f"{p['note'].upper()}\n{p['detail'].replace(chr(10), ' ')}\n"
              f"φ = {p['phi']:.2f}     D = {p['D']:.2f}  {p['ref']}")
    ax.text(0.5, -0.06, banner, transform=ax.transAxes, ha="center", va="top",
            fontsize=11.5, color=p["accent"], fontweight="bold", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=p["accent"], lw=1.8))


def main():
    masks = [grab_mask(p) for p in PANELS]
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.2, 0.9], hspace=0.72, wspace=0.28)
    for i, (p, m) in enumerate(zip(PANELS, masks)):
        draw_panel(fig.add_subplot(gs[0, i]), m, p)

    # middle transition arrow (figure coords)
    fig.patches.append(FancyArrowPatch((0.47, 0.78), (0.53, 0.78),
                       transform=fig.transFigure, arrowstyle="-|>",
                       mutation_scale=26, lw=2.5, color="0.35"))
    fig.text(0.5, 0.66, "more CuSO₄\n+ electric field\n→ migration &\nconvection\n→ weaker\nscreening",
             ha="center", va="center", fontsize=10, color="0.3", style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="0.7"))

    # bottom: the D scale, where our deposits sit
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(1.6, 2.03); ax.set_ylim(0, 1)
    ax.axvspan(1.71, 2.0, color="0.88", alpha=0.6)
    ax.axvline(1.71, color="k", ls="--", lw=1.6)
    ax.text(1.71, 1.10, "ideal 2D DLA (1.71)", ha="center", va="bottom", fontsize=10.5)
    ax.axvline(2.0, color="0.5", ls=":", lw=1.6)
    ax.text(2.0, 1.10, "compact (2.0)", ha="center", va="bottom", fontsize=10.5)
    # our points; 0.06 & 0.15 coincide at 1.87 -> merge
    pts = [(1.69, "#1f77b4", "0.02%"), (1.87, "#c0392b", "0.06 & 0.15%"),
           (1.91, "#7a2b2b", "0.04%")]
    for d, col, lab in pts:
        ax.plot(d, 0.5, "o", ms=14, color=col, zorder=5)
        ax.annotate(lab, (d, 0.5), textcoords="offset points", xytext=(0, -26),
                    ha="center", fontsize=10, color=col, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("measured effective fractal dimension  D", fontsize=12.5)
    ax.set_title("all our deposits sit between ideal DLA and compact — none at 1.71 except the most dilute",
                 fontsize=11.5, pad=26)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Why our fractal dimension exceeds the ideal DLA value (1.71)",
                 fontsize=16.5, fontweight="bold", y=0.995)
    out = FIGS / "screening_DLA_vs_dense.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"),
                                                                bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
