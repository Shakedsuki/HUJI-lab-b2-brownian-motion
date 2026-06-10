"""
radius_grad.py <run>
--------------------
Human-in-the-loop radius measurement on the STEEPEST-GRADIENT edge convention.

The accuracy bottleneck for k_B is the radius (k_B is linear in r). The automatic
outer-ring fit over-reads the true sphere (diffraction ring width); the previous
hand convention (bright-core boundary) still ran ~30% high. Here you size each
bead to the steepest intensity-gradient edge -- the classic sub-pixel geometric
edge, which sits closer to the true sphere edge -- with the auto gradient pick
shown as a GUIDE you confirm or override. The radial-intensity-profile panel
makes the edge visible so your clicks stay consistent bead-to-bead.

One bead at a time, shown at its SHARPEST frame (max FRST sym). The left panel is
the bead crop with the gradient guide (cyan dashed) and your circle (red); the
right panel is the azimuthal radial profile with the gradient edge (cyan) and
your radius (red) marked.

  left panel click .... move circle CENTER (gradient guide recomputes)
  right panel click ... set your radius to that point on the profile
  scroll up/down ...... radius +/- 0.5 px
  up / down arrow ..... radius +/- 0.1 px (fine)
  g ................... snap your radius to the gradient guide (current center)
  n / enter / right ... accept, next bead
  b / left ........... previous bead
  x ................... toggle bead BAD (not a clean single -> excluded)
  s ................... SAVE -> radius_manual.csv (+ labels.csv) then close
  q ................... quit without saving

Writes (for kept, non-bad beads):
  radius_manual.csv  (particle, r_px_manual, r_um_manual, method='gradient-edge')
  labels.csv         (particle, keep=1)
Then:  python analyze_run.py <run>      (prefers radius_manual.csv)
"""
import os
import sys

import numpy as np
import pandas as pd

from pipeline import paths, frames as fr, curate, shape


def _beadlist(out):
    """Curated singles (labels.csv keep==1 if present, else auto proposal),
    smallest first so the size trend is easy to scan."""
    cur = pd.read_csv(os.path.join(out, "curation.csv"))
    kept = curate.kept_pids(out)
    cur = (cur[cur["particle"].isin(kept)] if kept is not None
           else cur[cur["proposed"] == "single"])
    return cur.sort_values("R_px_med")


def _load_beads(stem):
    """Build the per-bead crops at each bead's sharpest frame + the gradient
    guide. Returns (beads, mpp). Headless-safe (no pyplot)."""
    import cv2
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
        r_auto = float(r["R_px_med"])
        h = int(max(4 * r_auto, 40))
        H, W = img.shape
        x, y = float(row["x_raw"]), float(row["y_raw"])
        xi, yi = int(round(x)), int(round(y))
        a, b = max(0, yi - h), max(0, xi - h)
        crop = img[a:min(H, yi + h), b:min(W, xi + h)]
        cx, cy = x - b, y - a
        r_grad_px, rs, prof = shape.gradient_edge_radius(crop, cx, cy)
        r0 = r_grad_px if np.isfinite(r_grad_px) else r_auto   # fallback
        beads.append(dict(pid=pid, crop=crop, cx=cx, cy=cy, r=float(r0),
                          r_grad=r_grad_px, r_auto=r_auto, rs=rs, prof=prof,
                          bad=False, tagged=np.isfinite(r_grad_px)))
    cap.release()
    return beads, mpp


def review(stem):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    out = paths.out_dir(stem)
    beads, mpp = _load_beads(stem)
    if not beads:
        print(f"[radius_grad] {stem}: no beads to measure")
        return

    state = {"i": 0}
    fig, (axI, axP) = plt.subplots(1, 2, figsize=(12.5, 6.4),
                                   gridspec_kw={"width_ratios": [1, 1]})

    def recompute_guide(bd):
        r_grad_px, rs, prof = shape.gradient_edge_radius(bd["crop"], bd["cx"], bd["cy"])
        bd["r_grad"], bd["rs"], bd["prof"] = r_grad_px, rs, prof

    def show():
        bd = beads[state["i"]]
        axI.clear(); axP.clear()
        axI.imshow(bd["crop"], cmap="gray")
        if np.isfinite(bd["r_grad"]):
            axI.add_patch(Circle((bd["cx"], bd["cy"]), bd["r_grad"], fill=False,
                                 edgecolor="#22e0e0", lw=1.2, ls="--"))
        axI.add_patch(Circle((bd["cx"], bd["cy"]), bd["r"], fill=False,
                             edgecolor="#ff3030", lw=1.8))
        axI.plot(bd["cx"], bd["cy"], "+", color="#ff3030", ms=9)
        axI.set_xticks([]); axI.set_yticks([])
        tag = "BAD" if bd["bad"] else ("set" if bd["tagged"] else "unset")
        ntag = sum(b["tagged"] and not b["bad"] for b in beads)
        gtxt = (f"{bd['r_grad'] * mpp:.3f}" if np.isfinite(bd["r_grad"]) else "n/a")
        axI.set_title(
            f"bead {state['i'] + 1}/{len(beads)}  p{bd['pid']}  [{tag}]  "
            f"({ntag} kept)\nr = {bd['r']:.1f}px = {bd['r'] * mpp:.3f}um   "
            f"(gradient guide {gtxt}um, auto-ring {bd['r_auto'] * mpp:.3f}um)",
            fontsize=10)
        # radial profile panel
        rs, prof = bd["rs"], bd["prof"]
        axP.plot(rs * mpp, prof, "-", color="0.25", lw=1.3)
        if np.isfinite(bd["r_grad"]):
            axP.axvline(bd["r_grad"] * mpp, color="#22a0a0", lw=1.4, ls="--",
                        label="gradient edge")
        axP.axvline(bd["r"] * mpp, color="#ff3030", lw=1.8, label="your radius")
        axP.set_xlabel(r"radius [$\mu$m]"); axP.set_ylabel("azimuthal intensity")
        axP.legend(fontsize=8, loc="best")
        axP.set_title("radial profile  (click here to set radius)", fontsize=10)
        fig.suptitle("left-click=center  right-panel click=radius  scroll/arrows="
                     "size  g=snap guide  n/b=next/prev  x=bad  s=SAVE  q=quit",
                     fontsize=9)
        fig.canvas.draw_idle()

    def on_click(ev):
        bd = beads[state["i"]]
        if ev.inaxes is axI and ev.xdata is not None:
            bd["cx"], bd["cy"] = ev.xdata, ev.ydata     # move center
            recompute_guide(bd)
            bd["tagged"] = True
            show()
        elif ev.inaxes is axP and ev.xdata is not None:
            bd["r"] = max(1.0, ev.xdata / mpp)          # click profile -> radius
            bd["tagged"] = True
            show()

    def on_scroll(ev):
        bd = beads[state["i"]]
        bd["r"] = max(1.0, bd["r"] + (0.5 if ev.button == "up" else -0.5))
        bd["tagged"] = True
        show()

    def nxt(d):
        state["i"] = (state["i"] + d) % len(beads)
        show()

    def save():
        rows = [dict(particle=b["pid"], r_px_manual=round(b["r"], 2),
                     r_um_manual=round(b["r"] * mpp, 4), method="gradient-edge")
                for b in beads if b["tagged"] and not b["bad"]]
        man = pd.DataFrame(rows)
        man.to_csv(os.path.join(out, "radius_manual.csv"), index=False)
        pd.DataFrame({"particle": man["particle"], "keep": 1}).to_csv(
            os.path.join(out, "labels.csv"), index=False)
        print(f"[radius_grad] {stem}: wrote radius_manual.csv + labels.csv -- "
              f"{len(rows)} beads -> {out}\n  now: python analyze_run.py {stem}")

    def on_key(ev):
        bd = beads[state["i"]]
        if ev.key in ("n", "enter", " ", "right"):
            bd["tagged"] = True; nxt(1)
        elif ev.key in ("b", "left"):
            nxt(-1)
        elif ev.key == "g" and np.isfinite(bd["r_grad"]):
            bd["r"] = bd["r_grad"]; bd["tagged"] = True; show()
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

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()


def dry_run(stem):
    """Headless check: load beads + guides, print a summary, write nothing."""
    beads, mpp = _load_beads(stem)
    n_guide = sum(np.isfinite(b["r_grad"]) for b in beads)
    print(f"[radius_grad:dry] {stem}: {len(beads)} curated beads loaded; "
          f"gradient guide available for {n_guide}/{len(beads)} "
          f"(rest fall back to auto-ring, flagged 'unset' for manual placement)")
    for b in beads[:8]:
        g = f"{b['r_grad'] * mpp:.3f}" if np.isfinite(b["r_grad"]) else "n/a"
        print(f"   p{b['pid']:<5} crop{b['crop'].shape} guide={g}um "
              f"auto-ring={b['r_auto'] * mpp:.3f}um")
    return beads


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gradient-edge radius GUI.")
    ap.add_argument("run", nargs="?", default="run7")
    ap.add_argument("--dry-run", action="store_true",
                    help="headless load+guide check (no window, writes nothing)")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args.run)
    else:
        review(args.run)
