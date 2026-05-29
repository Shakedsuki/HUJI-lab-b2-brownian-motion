"""
measure_radius.py  (week1-system-calibration)
---------------------------------------------
Measure each bead's PHYSICAL radius from the image, for Plot 2 (D vs 1/r).

Method: OUTER-EDGE points + robust CIRCLE FIT
---------------------------------------------
A bead in bright field is a bright core + a dark annulus; the true edge is the
OUTER boundary of that annulus. Earlier attempts failed two ways:
  - taking the intensity MINIMUM gave the middle of the dark ring -> radius too
    small (underestimate);
  - casting rays from trackpy's centroid (the bright CORE) fails when the core
    is not concentric with the ring (defocus/asymmetry) -> off-centre garbage.

Both are fixed by not trusting the seed centre and not using the minimum:
  1. along each of n_ang rays from the seed, find the dark-ring minimum (within
     a physical window [r_lo, r_hi]) then the OUTER edge = where I(r) recovers
     halfway from the ring minimum back to background;
  2. FIT A CIRCLE (robust, with outlier rejection) to those edge points.

The circle fit recovers the true CENTRE from the edge points (so an off-centre
seed self-corrects -> fixes p1/p7), the true OUTER RADIUS (fixes the
underestimate), and its RESIDUAL/R is a real roundness gate: small for a clean
sphere, large for a doublet/bean/blob. Robust rejection drops a doublet
partner's few off-circle points, recovering the MAIN sphere.

Outputs radius.csv with:
  r_px_med, r_um            : radius (median over frames)
  circ_resid_frac           : fit residual / R  -> ROUNDNESS (HIGH => exclude)
  inlier_frac               : fraction of rays on the circle (LOW => exclude)
  r_px_frame_cv, n_meas
Always eyeball radius_check.png (edge points + fitted circle per bead).

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/measure_radius.py run3 --tag d21m600
    python scripts/measure_radius.py run3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

import _paths


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


def _smooth(v, k):
    if k <= 0:
        return v
    w = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(v, w, mode="same")


def refine_center(crop, cx, cy, r_win, dark_pct=12):
    """Recentre on the dark-ring centroid in a window around (cx,cy).

    trackpy centres on the bright CORE, which can sit off the ring (defocus/
    asymmetry). The dark ring is the darkest feature; its darkness-weighted
    centroid is the ring centre. Re-seeding here means rays actually cross the
    bead from all directions (fixes the off-centre p1/p7 failure).
    """
    h, w = crop.shape
    x0 = max(0, int(cx - r_win)); x1 = min(w, int(cx + r_win) + 1)
    y0 = max(0, int(cy - r_win)); y1 = min(h, int(cy + r_win) + 1)
    sub = crop[y0:y1, x0:x1]
    thr = np.percentile(sub, dark_pct)
    m = sub <= thr
    if m.sum() < 5:
        return cx, cy
    ys, xs = np.nonzero(m)
    wts = (thr - sub[m]) + 1e-6                     # weight by darkness
    return x0 + float(np.average(xs, weights=wts)), y0 + float(np.average(ys, weights=wts))


def edge_points(crop, cx, cy, rmax, n_ang, smooth, r_lo, r_hi):
    """Outer-edge point on each ray = halfway recovery PAST the dark-ring min."""
    rs = np.arange(1.0, rmax, 0.5)
    win = (rs >= r_lo) & (rs <= r_hi)
    outer_band = rs > r_hi
    if not win.any():
        return np.empty((0, 2)), rs, None
    widx = np.where(win)[0]
    pts, prof_acc = [], []
    for th in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
        xs = cx + rs * np.cos(th)
        ys = cy + rs * np.sin(th)
        I = _smooth(map_coordinates(crop, [ys, xs], order=1, mode="nearest"), smooth)
        prof_acc.append(I)
        jmin = int(widx[np.argmin(I[win])])          # dark-ring minimum
        Imin = I[jmin]
        bg = np.median(I[outer_band]) if outer_band.sum() >= 2 else float(np.max(I[jmin:]))
        if bg <= Imin:
            continue
        thr = Imin + 0.5 * (bg - Imin)                # half-recovery toward background
        seg = I[jmin:]
        above = np.where(seg >= thr)[0]
        if len(above) == 0:
            continue
        r_edge = rs[jmin + above[0]]
        pts.append((cx + r_edge * np.cos(th), cy + r_edge * np.sin(th)))
    return np.array(pts), rs, (np.mean(prof_acc, axis=0) if prof_acc else None)


def _circle_fit(xy):
    """Algebraic (Kasa) circle fit -> (xc, yc, R)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0] / 2, sol[1] / 2
    R = float(np.sqrt(max(sol[2] + xc * xc + yc * yc, 0.0)))
    return float(xc), float(yc), R


def robust_circle(xy, iters=3, k=3.0):
    """Circle fit with MAD-based outlier rejection (drops doublet-partner pts)."""
    if len(xy) < 6:
        return None
    keep = np.ones(len(xy), bool)
    for _ in range(iters):
        xc, yc, R = _circle_fit(xy[keep])
        d = np.abs(np.hypot(xy[:, 0] - xc, xy[:, 1] - yc) - R)
        med = np.median(d[keep])
        mad = np.median(np.abs(d[keep] - med)) + 1e-6
        new = d < med + k * 1.4826 * mad
        if new.sum() < 6:
            break
        keep = new
    xc, yc, R = _circle_fit(xy[keep])
    d = np.abs(np.hypot(xy[:, 0] - xc, xy[:, 1] - yc) - R)
    resid = float(np.sqrt(np.mean(d[keep] ** 2)))
    return xc, yc, R, resid, keep


def main():
    ap = argparse.ArgumentParser(description="Bead radius via outer-edge circle fit.")
    ap.add_argument("run", help="run stem, e.g. run3")
    ap.add_argument("--tag", default=None, help="measurements/<run>/<tag>/")
    ap.add_argument("--beads", type=int, nargs="*", default=None,
                    help="restrict to these particle ids (e.g. the real spheres you picked); "
                         "default = all beads with enough frames")
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--n-frames", type=int, default=9)
    ap.add_argument("--half", type=int, default=25, help="crop half-width (px)")
    ap.add_argument("--n-ang", type=int, default=36, help="rays per bead")
    ap.add_argument("--smooth", type=int, default=2)
    ap.add_argument("--r-lo", type=float, default=2.0, help="min ring-min radius (px)")
    ap.add_argument("--r-hi", type=float, default=18.0, help="max ring-min radius (px)")
    args = ap.parse_args()

    import cv2

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    tcsv = os.path.join(cdir, "trajectory.csv")
    if not os.path.exists(tcsv):
        sys.exit(f"no trajectory.csv in {cdir} -- run track.py first")

    mpp = _paths.load_scale() or 1.0
    meta_video = _paths.load_runs().get("runs", {}).get(stem, {}).get("video", stem + ".avi")
    path = _paths.video(meta_video)
    traj = pd.read_csv(tcsv)
    counts = traj.groupby("particle")["frame"].count()
    pids = counts[counts >= args.min_len].index.tolist()
    if args.beads:
        want = set(args.beads)
        pids = [p for p in pids if p in want]
    print(f"[radius] {stem}: {len(pids)} beads (>= {args.min_len} frames), mpp={mpp} um/px")

    cap = cv2.VideoCapture(path)
    H = args.half
    rows, gallery = [], []
    for pid in pids:
        sub = traj[traj["particle"] == pid].sort_values("frame")
        idxs = np.linspace(0, len(sub) - 1, args.n_frames).astype(int)
        Rs, resids, infracs, disp = [], [], [], None
        for _, row in sub.iloc[idxs].iterrows():
            fr = gray_frame(cap, int(row["frame"]))
            if fr is None:
                continue
            x, y = float(row["x"]), float(row["y"])
            x0, y0 = int(round(x)) - H, int(round(y)) - H
            if x0 < 0 or y0 < 0 or x0 + 2 * H + 1 > fr.shape[1] or y0 + 2 * H + 1 > fr.shape[0]:
                continue
            crop = fr[y0:y0 + 2 * H + 1, x0:x0 + 2 * H + 1]
            cx, cy = x - x0, y - y0
            # re-seed onto the dark ring, then detect->fit->recentre twice so an
            # off-centre trackpy seed converges onto the actual bead.
            cx, cy = refine_center(crop, cx, cy, args.r_hi + 4)
            fit = None
            for _ in range(2):
                pts, rs, prof = edge_points(crop, cx, cy, H, args.n_ang,
                                            args.smooth, args.r_lo, args.r_hi)
                if len(pts) < 6:
                    break
                fit = robust_circle(pts)
                if fit is None:
                    break
                cx, cy = fit[0], fit[1]               # recentre on the fit, redo
            if fit is None:
                continue
            xc, yc, R, resid, keep = fit
            Rs.append(R)
            resids.append(resid / R if R else np.nan)
            infracs.append(float(keep.mean()))
            if disp is None:
                disp = (crop, pts, keep, xc, yc, R, rs, prof)
        if len(Rs) < 3:
            continue
        Rs = np.array(Rs)
        R_med = float(np.median(Rs))
        rows.append({
            "particle": int(pid),
            "r_px_med": R_med,
            "r_um": R_med * mpp,
            "circ_resid_frac": float(np.median(resids)),   # roundness (HIGH=>exclude)
            "inlier_frac": float(np.median(infracs)),       # LOW => not a circle
            "r_px_frame_cv": float(Rs.std() / Rs.mean()) if Rs.mean() else np.nan,
            "n_meas": int(len(Rs)),
        })
        if len(gallery) < 9 and disp is not None:
            gallery.append((pid, R_med, float(np.median(resids)), *disp))
    cap.release()

    if not rows:
        sys.exit("[radius] measured nothing -- check trajectory/video paths")
    out = pd.DataFrame(rows).sort_values("r_px_med", ascending=False)
    out.to_csv(os.path.join(cdir, "radius.csv"), index=False)
    print(f"[radius] wrote radius.csv ({len(out)} beads) -> {cdir}")
    clean = out[(out.circ_resid_frac < 0.10) & (out.inlier_frac > 0.6)]
    print(f"[radius] {len(clean)}/{len(out)} pass gates (circ_resid_frac<0.10, inlier_frac>0.6)")
    if len(clean):
        print(f"[radius] clean r_um range {clean.r_um.min():.2f} - {clean.r_um.max():.2f} um")

    # ---- verification montage --------------------------------------------
    n = len(gallery)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8), squeeze=False)
    th = np.linspace(0, 2 * np.pi, 120)
    for j, (pid, R, rf, crop, pts, keep, xc, yc, Rfit, rs, prof) in enumerate(gallery):
        ax = axes[0][j]
        ax.imshow(crop, cmap="gray")
        if len(pts):
            ax.scatter(pts[keep, 0], pts[keep, 1], s=7, c="yellow", zorder=3)     # inliers
            if (~keep).any():
                ax.scatter(pts[~keep, 0], pts[~keep, 1], s=7, c="cyan", zorder=3)  # dropped
        ax.plot(xc + Rfit * np.cos(th), yc + Rfit * np.sin(th), "r-", lw=1.3)
        ax.plot(xc, yc, "r+", ms=7)
        flag = "" if rf < 0.10 else "  REJECT"
        ax.set_title(f"p{pid}  r={R:.1f}px\nresid/R={rf:.2f}{flag}", fontsize=8)
        ax.axis("off")
        ax = axes[1][j]
        if prof is not None:
            ax.plot(rs, prof, lw=1)
        ax.axvspan(args.r_lo, args.r_hi, color="C2", alpha=0.08)
        ax.axvline(R, color="r", ls="--", lw=1)
        ax.set_xlabel("r [px]", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel("mean I(r)", fontsize=7)
    fig.suptitle(f"{stem}: outer-edge circle fit. yellow=edge pts (inliers), cyan=dropped, "
                 f"red=fitted circle. resid/R=roundness (>0.10 => reject)")
    fig.tight_layout()
    fig.savefig(os.path.join(cdir, "radius_check.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[radius] wrote radius_check.png -> {cdir}  (EYEBALL THIS)")


if __name__ == "__main__":
    main()
