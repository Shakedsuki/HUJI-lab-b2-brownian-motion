"""
shape.py  (pipeline)
--------------------
Per-bead shape analysis in an image crop. Shared by curation (singleton vs
doublet/blob) and the radius stage (physical radius). All metrics are computed
from the OUTER EDGE of the dark diffraction ring + the core, with explicit
handling of both focus polarities.

measure_shape() returns, for one crop centred near a detection:
  xc, yc            refined centre (full-image px) from the circle fit
  R                 outer-edge radius (px)   -> physical size (the ring's OUTER
                    boundary; note the diffraction bias handled in radius.py)
  circ_resid_frac   RMS circle-fit residual / R  -> ROUNDNESS (high => not round)
  inlier_frac       fraction of rays whose edge point lies on the circle
  ring_cv           angular CV of per-ray edge radius -> asymmetry / contact-neck
  n_cores           count of core extrema inside -> 2 => doublet (the direct test)
  ecc               intensity-moment eccentricity -> elongation (doublet/merged)

Why outer-edge + robust circle fit (not centroid / threshold): a bright-field
bead is a bright/dark core + dark ring. The intensity MINIMUM is the middle of
the ring (radius too small); the centroid is the bright core which can sit off
the ring centre. Casting rays to the ring's outer half-recovery edge and fitting
a circle with outlier rejection recovers the true centre (off-centre cores
self-correct) and the true outer radius, and the fit residual is a real
roundness gate that a doublet/blob fails.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates


def _kasa(xy):
    """Algebraic (Kasa) circle fit -> (xc, yc, R)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0] / 2, sol[1] / 2
    R = float(np.sqrt(max(sol[2] + xc * xc + yc * yc, 0.0)))
    return float(xc), float(yc), R


def _robust_circle(xy, iters=3, k=2.5):
    """Kasa fit with MAD outlier rejection (drops doublet-partner / blob points)."""
    if len(xy) < 6:
        return None
    keep = np.ones(len(xy), bool)
    for _ in range(iters):
        xc, yc, R = _kasa(xy[keep])
        d = np.abs(np.hypot(xy[:, 0] - xc, xy[:, 1] - yc) - R)
        med = np.median(d[keep])
        mad = np.median(np.abs(d[keep] - med)) + 1e-6
        new = d < med + k * 1.4826 * mad
        if new.sum() < 6:
            break
        keep = new
    xc, yc, R = _kasa(xy[keep])
    d = np.abs(np.hypot(xy[:, 0] - xc, xy[:, 1] - yc) - R)
    resid = float(np.sqrt(np.mean(d[keep] ** 2)))
    return xc, yc, R, resid, keep


def _ray_edges(crop, cx, cy, rmax, polarity, n_ang=48, r_lo=2.0, smooth=1.0,
               recover_px=8.0):
    """Edge point on each ray = the PRIMARY (innermost) diffraction ring's outer
    side. Scan outward to the first SIGNIFICANT ring extremum, then take the
    half-recovery toward the LOCAL recovery peak within a bounded window
    (~recover_px). Locking to the first ring -- instead of recovering all the way
    to far background -- stops the radius over-shooting onto an outer halo on
    multi-ring / defocused beads, while reducing to the plain outer edge for a
    clean single-ring bead (whose recovery peak IS the background).
    Polarity-aware via v = sgn*I, so the ring is always a MIN of v."""
    rs = np.arange(1.0, rmax, 0.5)
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    sgn = 1.0 if polarity >= 0 else -1.0
    window = int(round(recover_px / 0.5))
    pts, edge_r, prof_acc = [], [], []
    for t in th:
        xs = cx + rs * np.cos(t)
        ys = cy + rs * np.sin(t)
        I = map_coordinates(crop, [ys, xs], order=1, mode="nearest")
        if smooth:
            kk = max(int(smooth), 1)
            I = np.convolve(I, np.ones(2 * kk + 1) / (2 * kk + 1), mode="same")
        prof_acc.append(I)
        v = sgn * I                                      # ring is a MIN of v
        widx = np.where(rs >= r_lo)[0]
        if len(widx) < 3:
            continue
        spread = float(np.ptp(v[widx])) + 1e-9
        # primary ring = first significant local min scanning outward
        jext, run_max = None, v[widx[0]]
        for k in widx[1:-1]:
            run_max = max(run_max, v[k])
            if (v[k] <= v[k - 1] and v[k] <= v[k + 1]
                    and run_max - v[k] > 0.10 * spread):
                jext = k
                break
        if jext is None:
            jext = int(widx[np.argmin(v[widx])])
        hi = min(jext + window, len(rs) - 1)
        seg = v[jext:hi + 1]
        loc_max = float(seg.max())
        if loc_max - v[jext] <= 0:                       # no recovery on this ray
            continue
        thr = v[jext] + 0.5 * (loc_max - v[jext])
        above = np.where(seg >= thr)[0]
        if len(above) == 0:
            continue
        a = int(above[0])
        if a > 0 and seg[a] != seg[a - 1]:               # sub-pixel crossing
            frac = np.clip((thr - seg[a - 1]) / (seg[a] - seg[a - 1]), 0.0, 1.0)
            re = rs[jext + a - 1] + frac * (rs[jext + a] - rs[jext + a - 1])
        else:
            re = rs[jext + a]
        pts.append((cx + re * np.cos(t), cy + re * np.sin(t)))
        edge_r.append(re)
    prof = np.mean(prof_acc, axis=0) if prof_acc else None
    return np.array(pts), np.array(edge_r), rs, prof


def _n_cores(crop, cx, cy, R, polarity):
    """Count distinct core extrema inside r<0.65R. 1 = single, >=2 = doublet."""
    if R <= 2:
        return 1
    s = gaussian_filter(crop * (1.0 if polarity >= 0 else -1.0), 1.0)
    H, W = s.shape
    yy, xx = np.ogrid[:H, :W]
    inner = np.hypot(xx - cx, yy - cy) <= 0.65 * R
    if inner.sum() < 4:
        return 1
    foot = max(3, int(round(0.6 * R)) | 1)              # odd footprint ~ core sep
    mx = maximum_filter(s, size=foot)
    lo = np.median(s[inner])
    hi = s[inner].max()
    if hi <= lo:
        return 1
    thr = lo + 0.45 * (hi - lo)                         # prominence cut
    peaks = inner & (s == mx) & (s >= thr)
    return int(min(peaks.sum(), 4)) or 1


def _edge_ecc(pts, xc, yc):
    """Eccentricity of the EDGE-POINT cloud about the fitted centre. Points on a
    true circle are isotropic -> ecc~0; a doublet/blob's peanut outline is
    elongated -> ecc high. Cleaner + more direct than intensity moments (which
    are biased by core brightness structure)."""
    if pts is None or len(pts) < 5:
        return np.nan
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - yc
    cxx = float(np.mean(dx * dx))
    cyy = float(np.mean(dy * dy))
    cxy = float(np.mean(dx * dy))
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = max(tr * tr / 4 - det, 0.0)
    l1 = tr / 2 + np.sqrt(disc)
    l2 = tr / 2 - np.sqrt(disc)
    return float(np.sqrt(max(1 - l2 / l1, 0.0))) if l1 > 0 else np.nan


def measure_shape(img, x, y, r_seed, polarity=1, half=None, n_ang=48,
                  r_lo=2.0, smooth=1.0, recenter_iter=2):
    """Full shape measurement around (x, y) in flat-fielded image `img`."""
    H, W = img.shape
    half = int(half if half is not None else max(2.0 * r_seed + 10, 18))
    xi, yi = int(round(x)), int(round(y))
    x0, y0 = max(0, xi - half), max(0, yi - half)
    x1, y1 = min(W, xi + half + 1), min(H, yi + half + 1)
    crop = img[y0:y1, x0:x1]
    cx, cy = x - x0, y - y0
    rmax = min(half, crop.shape[0] / 2 - 1, crop.shape[1] / 2 - 1)

    fail = dict(xc=x, yc=y, R=np.nan, circ_resid_frac=np.nan, inlier_frac=0.0,
                ring_cv=np.nan, n_cores=0, ecc=np.nan,
                _edge=None, _keep=None, _prof=None, _rs=None, _crop=crop,
                _cxy=(cx, cy))
    if rmax < r_lo + 2:
        return fail

    fit = None
    edge_r = None
    for _ in range(max(1, recenter_iter)):
        pts, edge_r, rs, prof = _ray_edges(crop, cx, cy, rmax, polarity,
                                           n_ang, r_lo, smooth)
        if len(pts) < 6:
            return fail
        fit = _robust_circle(pts)
        if fit is None:
            return fail
        cx, cy = fit[0], fit[1]
    xc, yc, R, resid, keep = fit
    # edge radius spread about the FINAL centre (asymmetry / contact-neck)
    er = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
    ring_cv = float(er.std() / er.mean()) if er.mean() > 0 else np.nan

    return dict(
        xc=x0 + xc, yc=y0 + yc, R=float(R),
        circ_resid_frac=float(resid / R) if R else np.nan,
        inlier_frac=float(keep.mean()),
        ring_cv=ring_cv,
        n_cores=_n_cores(crop, xc, yc, R, polarity),
        ecc=_edge_ecc(pts, xc, yc),
        _edge=pts, _keep=keep, _prof=prof, _rs=rs, _crop=crop, _cxy=(xc, yc),
    )


if __name__ == "__main__":   # python -m pipeline.shape run3 --frame 0
    import argparse
    import os
    import matplotlib.pyplot as plt
    from . import paths, frames as fr, figstyle, detect

    ap = argparse.ArgumentParser(description="Shape-metric montage on one frame.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()

    figstyle.set_style()
    vid = paths.video(paths.video_for_run(args.run), args.videos_dir)
    flat = fr.flat_field(vid, n_sample=40)
    img = fr.frame_at(vid, args.frame) - flat
    d = detect.detect_frame(img)
    order = np.argsort(d["sym"])[::-1][:args.n]

    ncol = 6
    nrow = int(np.ceil(len(order) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow), squeeze=False)
    th = np.linspace(0, 2 * np.pi, 120)
    for k, idx in enumerate(order):
        m = measure_shape(img, d["x"][idx], d["y"][idx], d["r_est"][idx],
                          d["polarity"][idx])
        ax = axs[k // ncol][k % ncol]
        ax.imshow(m["_crop"], cmap="gray")
        cxy = m["_cxy"]
        if m["_edge"] is not None and len(m["_edge"]):
            e, kp = m["_edge"], m["_keep"]
            ax.scatter(e[kp, 0], e[kp, 1], s=6, c="yellow")
            if (~kp).any():
                ax.scatter(e[~kp, 0], e[~kp, 1], s=6, c="cyan")
        if np.isfinite(m["R"]):
            ax.plot(cxy[0] + m["R"] * np.cos(th), cxy[1] + m["R"] * np.sin(th),
                    "r-", lw=1.2)
        ax.plot(cxy[0], cxy[1], "r+", ms=8)
        ax.set_title(f"R={m['R']:.1f} res={m['circ_resid_frac']:.02f}\n"
                     f"cores={m['n_cores']} ecc={m['ecc']:.02f} "
                     f"rcv={m['ring_cv']:.02f}", fontsize=7)
        ax.axis("off")
        print(f"  bead{k:2d}: xy=({d['x'][idx]:.0f},{d['y'][idx]:.0f}) "
              f"R={m['R']:5.1f}px resid/R={m['circ_resid_frac']:.3f} "
              f"inlier={m['inlier_frac']:.2f} ring_cv={m['ring_cv']:.3f} "
              f"n_cores={m['n_cores']} ecc={m['ecc']:.3f}")
    for k in range(len(order), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{args.run} f{args.frame}: shape metrics "
                 f"(yellow=edge inliers, cyan=dropped, red=fit)")
    tdir = os.path.join(paths.out_dir(args.run), "shape_tune")
    p = figstyle.save(fig, os.path.join(tdir, f"shape_f{args.frame}.png"))
    plt.close(fig)
    print(f"[shape-tune] wrote {p}")
