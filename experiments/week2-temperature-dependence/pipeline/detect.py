"""
detect.py  (pipeline)
---------------------
Fast Radial Symmetry Transform (FRST) detector for bright-field beads.

WHY NOT trackpy.locate: Crocker-Grier expects filled Gaussian spots of one
fixed polarity (fluorescence). A bright-field bead is a bright/dark CORE ringed
by a dark diffraction annulus, and the core contrast FLIPS bright<->dark as the
bead moves through focus. A single-polarity blob finder mis-detects half of them
and gets confused by the ring.

FRST keys on RADIAL SYMMETRY of the intensity gradient instead: every gradient
vector votes for a center one radius n away (both directions), so a circular
feature -- whose gradients are all radial -- accumulates votes at its center
from all orientations, while straight edges (debris, doublet contact necks) and
the diffuse out-of-focus halos do not. It is polarity-free (we accumulate |votes|
so a dark or bright core both peak) and intrinsically circularity-preferring, so
debris/halos are suppressed at DETECTION rather than filtered afterward.

What this does NOT solve on its own: a doublet's two partners are each circular,
so FRST happily finds both centers. That is fine -- the rigid-pair temporal test
in curate.py removes them. FRST's job is to find well-localized circular centers
with a beadness score and a rough radius; curation decides singleton-ness.

Outputs per detection: frame, x, y, sym (symmetry/beadness score), r_est_px
(dominant symmetric radius -> seeds the radius stage + spans polydispersity),
polarity (+1 bright core / -1 dark core), contrast.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates

DEFAULT_RADII = np.arange(3, 21, 3)      # px voting radii; diam 6-40 px (~0.4-2.9 um)


def sobel_grad(img, presmooth=1.0):
    """Gradient (gx, gy, magnitude) of a lightly pre-smoothed image."""
    import cv2
    g = gaussian_filter(img, presmooth) if presmooth else img
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def frst(img, radii=DEFAULT_RADII, alpha=2.0, grad_pct=84.0, presmooth=1.0,
         smooth=1.5):
    """Polarity-free FRST symmetry map S (H,W); peaks at circular-feature centers.

    Speed-tuned: votes only from significant-gradient pixels (the rings; the flat
    background has ~zero gradient), per-radius normalization by max() (cheap, and
    no hard clip -> no NMS-defeating plateaus), and a SINGLE smoothing of the
    summed map instead of one per radius.
    """
    H, W = img.shape
    gx, gy, mag = sobel_grad(img, presmooth)
    thr = np.percentile(mag, grad_pct)
    ys, xs = np.nonzero(mag > thr)
    if len(xs) == 0:
        return np.zeros((H, W), np.float32)
    gm = mag[ys, xs].astype(np.float32)
    ux = (gx[ys, xs] / gm).astype(np.float32)
    uy = (gy[ys, xs] / gm).astype(np.float32)

    # +ve and -ve votes accumulated in one bincount each via signed weights
    w_o = np.concatenate([np.ones(len(xs), np.float32), -np.ones(len(xs), np.float32)])
    w_m = np.concatenate([gm, -gm])
    HW = H * W
    kk = max(int(0.001 * HW), 1)                        # ~99.9th percentile rank
    S = np.zeros(HW, np.float32)
    for n in radii:
        lp = (np.clip(np.round(ys + n * uy), 0, H - 1).astype(np.int64) * W
              + np.clip(np.round(xs + n * ux), 0, W - 1).astype(np.int64))
        ln = (np.clip(np.round(ys - n * uy), 0, H - 1).astype(np.int64) * W
              + np.clip(np.round(xs - n * ux), 0, W - 1).astype(np.int64))
        idx = np.concatenate([lp, ln])
        O = np.bincount(idx, weights=w_o, minlength=HW)
        M = np.bincount(idx, weights=w_m, minlength=HW)
        ka = max(np.abs(O).max(), 1.0)
        Fn = (np.abs(O) / ka) ** alpha * np.abs(M)
        # normalize by ~99.9th percentile (via O(n) partition, not a full sort)
        # so bead peaks stand far above background and add across radii.
        hi = np.partition(Fn, HW - kk)[HW - kk]
        if hi > 0:
            S += Fn / hi
    S = gaussian_filter(S.reshape(H, W), smooth)
    return (S / len(radii)).astype(np.float32)


def _subpix(S, x, y):
    """Parabolic sub-pixel refine of a local max at integer (x,y)."""
    H, W = S.shape
    dx = dy = 0.0
    if 0 < x < W - 1:
        a, b, c = S[y, x - 1], S[y, x], S[y, x + 1]
        d = a - 2 * b + c
        if d != 0:
            dx = 0.5 * (a - c) / d
    if 0 < y < H - 1:
        a, b, c = S[y - 1, x], S[y, x], S[y + 1, x]
        d = a - 2 * b + c
        if d != 0:
            dy = 0.5 * (a - c) / d
    return x + np.clip(dx, -1, 1), y + np.clip(dy, -1, 1)


def ring_radius_polarity(img, x, y, rmax=26, n_ang=16):
    """Estimate bead radius + core polarity from the azimuthally-averaged radial
    intensity profile about (x, y).

    The FRST per-pixel radius locks onto the bright/dark CORE (a sharp compact
    symmetry peak), not the dark diffraction RING that marks the bead edge -- so
    it is useless as a size seed. The radial profile gives the ring directly:
    a bright-core bead dips to a minimum at the dark ring; a dark-core bead peaks
    at its bright halo. Returns (r_est_px, polarity, contrast). This is a coarse
    SEED only; shape.py does the accurate outer-edge circle fit.
    """
    rs = np.arange(1.0, rmax, 1.0)
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    xs = x + np.cos(th)[:, None] * rs[None, :]
    ys = y + np.sin(th)[:, None] * rs[None, :]
    prof = map_coordinates(img, [ys.ravel(), xs.ravel()], order=1,
                           mode="nearest").reshape(n_ang, len(rs)).mean(0)
    core = float(prof[:2].mean())
    bg = float(np.median(prof[-4:]))
    pol = 1 if core >= bg else -1
    win = rs >= 2
    k = np.argmin(prof[win]) if pol >= 0 else np.argmax(prof[win])
    return float(rs[win][k]), int(pol), abs(core - bg)


def _ring_batch(img, X, Y, rmax=26, n_ang=16):
    """Vectorized ring radius + polarity for MANY centers in ONE map_coordinates
    call (per-detection calls dominated the per-frame cost otherwise)."""
    rs = np.arange(1.0, rmax, 1.0)
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    cos = np.cos(th)[None, :, None]
    sin = np.sin(th)[None, :, None]
    XX = X[:, None, None] + cos * rs[None, None, :]      # (nc, n_ang, n_r)
    YY = Y[:, None, None] + sin * rs[None, None, :]
    prof = map_coordinates(img, [YY.ravel(), XX.ravel()], order=1,
                           mode="nearest").reshape(len(X), n_ang, len(rs)).mean(1)
    core = prof[:, :2].mean(1)
    bg = np.median(prof[:, -4:], axis=1)
    pol = np.where(core >= bg, 1, -1)
    win = rs >= 2
    sub = prof[:, win]
    k = np.where(pol >= 0, np.argmin(sub, axis=1), np.argmax(sub, axis=1))
    r_est = rs[win][k]
    return r_est, pol, np.abs(core - bg)


def detect_frame(img, radii=DEFAULT_RADII, alpha=2.0, grad_pct=84.0,
                 presmooth=1.0, sym_min=0.18, min_sep=6, border=4, S=None):
    """Detect bead centers in one (flat-fielded) frame.

    Returns a dict of equal-length arrays: x, y, sym, r_est, polarity, contrast.
    """
    if S is None:
        S = frst(img, radii, alpha, grad_pct, presmooth)
    mx = maximum_filter(S, size=min_sep)
    peaks = (S == mx) & (S >= sym_min)
    peaks[:border] = peaks[-border:] = peaks[:, :border] = peaks[:, -border:] = False
    ys, xs = np.nonzero(peaks)
    # greedy min-distance NMS: accept strongest first, reject anything within
    # min_sep of an accepted peak. Dedupes flat-topped plateaus and double hits
    # on one bead, while keeping genuinely separate (e.g. doublet-partner) peaks.
    vals = S[ys, xs]
    order = np.argsort(vals)[::-1]
    ax, ay = [], []
    sep2 = min_sep * min_sep
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all((x - jx) ** 2 + (y - jy) ** 2 >= sep2 for jx, jy in zip(ax, ay)):
            ax.append(x)
            ay.append(y)
    keys = ("x", "y", "sym", "r_est", "polarity", "contrast")
    if not ax:
        return {k: [] for k in keys}
    ax = np.asarray(ax); ay = np.asarray(ay)
    xsp = np.empty(len(ax)); ysp = np.empty(len(ax))
    for i in range(len(ax)):
        xsp[i], ysp[i] = _subpix(S, int(ax[i]), int(ay[i]))
    rmax = int(max(radii) + 6)
    r_est, pol, con = _ring_batch(img, xsp, ysp, rmax)
    return dict(x=xsp, y=ysp, sym=S[ay, ax], r_est=r_est,
                polarity=pol, contrast=con)


def _detect_one(img, kw, downscale):
    """Flat-fielded full-res frame -> detections (full-res px), with optional
    downscale for the FRST accumulator."""
    import cv2
    if downscale > 1:
        img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale),
                         interpolation=cv2.INTER_AREA)
    d = detect_frame(img, **kw)
    if downscale > 1 and len(d["x"]):
        d["x"] = d["x"] * downscale
        d["y"] = d["y"] * downscale
        d["r_est"] = d["r_est"] * downscale
    return d


def _detect_range_worker(args):
    """Worker: detect a contiguous frame range [start, start+count). Each worker
    opens its own VideoCapture and reads sequentially from `start` (MJPEG is all-
    intra so the seek is exact). Module-level + picklable for spawn.

    Pins itself to ONE thread (cv2 + BLAS) -- otherwise N workers x M internal
    threads oversubscribe the cores and run SLOWER than serial."""
    import cv2
    import pandas as pd
    cv2.setNumThreads(1)
    video_path, flat, start, count, kw, downscale = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    parts = []
    for j in range(count):
        ok, frm = cap.read()
        if not ok:
            break
        img = np.asarray(frm)[..., :3].mean(-1).astype(np.float32)
        if flat is not None:
            img = img - flat
        d = _detect_one(img, kw, downscale)
        if len(d["x"]):
            df = pd.DataFrame(d)
            df["frame"] = int(start) + j
            parts.append(df)
    cap.release()
    return pd.concat(parts, ignore_index=True) if parts else None


def detect_clip(video_path, flat=None, radii=DEFAULT_RADII, alpha=2.0,
                grad_pct=84.0, presmooth=1.0, sym_min=0.18, min_sep=6,
                max_frames=None, progress=100, downscale=1, workers=1):
    """Stream a clip and return a features DataFrame for linking.

    Columns: frame, x, y, sym, r_est, polarity, contrast (positions in full-res
    PIXELS regardless of downscale).

    downscale>1: the FRST accumulator (cost ~ pixel count) runs on a 1/downscale
    frame, then positions/radii scale back. Coarser centres feed the MSD intercept
    c, not the slope 4D, so D is preserved at ~downscale^2 speed.

    workers>1: FRST is single-threaded and per-frame independent, so detection
    fans out across processes (each reads its own contiguous frame range). ~linear
    in cores -- the dominant batch-tracking speedup on a multi-core box."""
    import pandas as pd
    import cv2
    from . import frames as fr

    if downscale > 1:
        radii = np.unique(np.maximum(
            2, (np.asarray(radii) / downscale).round().astype(int)))
        min_sep = max(3, int(round(min_sep / downscale)))
    kw = dict(radii=radii, alpha=alpha, grad_pct=grad_pct, presmooth=presmooth,
              sym_min=sym_min, min_sep=min_sep)

    if workers and workers > 1:
        import os
        # children inherit these on spawn -> single-threaded BLAS per worker
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(_v, "1")
        from concurrent.futures import ProcessPoolExecutor
        total = fr.count_frames(video_path)
        total = min(total, max_frames) if (max_frames and total > 0) else (total or max_frames or 0)
        chunk = max(1, (total + workers - 1) // workers)
        tasks, s = [], 0
        while s < total:
            tasks.append((video_path, flat, s, min(chunk, total - s), kw, downscale))
            s += chunk
        print(f"    [detect] parallel: {total} frames / {len(tasks)} workers "
              f"(downscale={downscale})", flush=True)
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_detect_range_worker, tasks):
                if r is not None:
                    parts.append(r)
        if not parts:
            return pd.DataFrame(columns=["frame", "x", "y", "sym", "r_est",
                                         "polarity", "contrast"])
        return pd.concat(parts, ignore_index=True).sort_values(
            "frame").reset_index(drop=True)

    parts = []
    for i, frame in enumerate(fr.iter_frames(video_path, max_frames)):
        img = frame - flat if flat is not None else frame
        d = _detect_one(img, kw, downscale)
        if len(d["x"]):
            df = pd.DataFrame(d)
            df["frame"] = i
            parts.append(df)
        if progress and (i + 1) % progress == 0:
            n = len(parts[-1]) if parts else 0
            print(f"    [detect] frame {i + 1}: {n} features", flush=True)
    if not parts:
        return pd.DataFrame(columns=["frame", "x", "y", "sym", "r_est",
                                     "polarity", "contrast"])
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":   # tune: python -m pipeline.detect run3 --frame 0
    import argparse
    import os
    import matplotlib.pyplot as plt
    from . import paths, frames as fr, figstyle

    ap = argparse.ArgumentParser(description="FRST detector tuning on one frame.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--sym-min", type=float, default=0.18)
    ap.add_argument("--grad-pct", type=float, default=80.0)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()

    figstyle.set_style()
    vid = paths.video(paths.video_for_run(args.run), args.videos_dir)
    print(f"[detect-tune] {args.run} frame {args.frame}: flat-fielding...")
    flat = fr.flat_field(vid, n_sample=40)
    raw = fr.frame_at(vid, args.frame)
    img = raw - flat
    S = frst(img, alpha=args.alpha, grad_pct=args.grad_pct)
    d = detect_frame(img, alpha=args.alpha, grad_pct=args.grad_pct,
                     sym_min=args.sym_min, S=S)
    n = len(d["x"])
    print(f"[detect-tune] {n} detections; sym range "
          f"{min(d['sym']) if n else 0:.2f}-{max(d['sym']) if n else 0:.2f}; "
          f"S background p50={np.percentile(S,50):.3f} p99={np.percentile(S,99):.3f}")

    out = paths.out_dir(args.run)
    tdir = os.path.join(out, "detect_tune")
    os.makedirs(tdir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(raw, cmap="gray")
    th = np.linspace(0, 2 * np.pi, 60)
    for x, y, r, pol in zip(d["x"], d["y"], d["r_est"], d["polarity"]):
        c = "lime" if pol >= 0 else "cyan"
        ax[0].plot(x + r * np.cos(th), y + r * np.sin(th), "-", color=c, lw=0.8)
        ax[0].plot(x, y, "+", color=c, ms=5)
    ax[0].set_title(f"{args.run} f{args.frame}: {n} FRST detections "
                    f"(lime=bright core, cyan=dark core)")
    ax[0].axis("off")
    im = ax[1].imshow(S, cmap="magma")
    ax[1].set_title("FRST symmetry map S")
    ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    p = figstyle.save(fig, os.path.join(tdir, f"frst_f{args.frame}.png"))
    plt.close(fig)
    print(f"[detect-tune] wrote {p}")

    # zoom: native-resolution crop around the densest cluster of detections so
    # ring-centering + r_est can actually be eyeballed (the full frame downscales
    # too far). Pick the median detection location as the crop centre.
    if n:
        cx, cy = int(np.median(d["x"])), int(np.median(d["y"]))
        half = 260
        H, W = raw.shape
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(W, cx + half), min(H, cy + half)
        crop = raw[y0:y1, x0:x1]
        figz, axz = plt.subplots(figsize=(7, 7))
        axz.imshow(crop, cmap="gray")
        for x, y, r, pol, s in zip(d["x"], d["y"], d["r_est"], d["polarity"], d["sym"]):
            if x0 <= x < x1 and y0 <= y < y1:
                c = "lime" if pol >= 0 else "cyan"
                axz.plot((x - x0) + r * np.cos(th), (y - y0) + r * np.sin(th),
                         "-", color=c, lw=1.2)
                axz.plot(x - x0, y - y0, "+", color=c, ms=6)
                axz.text(x - x0 + r + 2, y - y0, f"{s:.2f}", color="yellow",
                         fontsize=6, va="center")
        axz.set_title(f"{args.run} f{args.frame} zoom @({cx},{cy}); circle=r_est, "
                      f"label=sym")
        axz.axis("off")
        pz = figstyle.save(figz, os.path.join(tdir, f"frst_f{args.frame}_zoom.png"))
        plt.close(figz)
        print(f"[detect-tune] wrote {pz}")

        # gallery: the strongest detections as individual crops, to eyeball
        # centering + r_est on the clearest beads.
        order = np.argsort(d["sym"])[::-1][:12]
        ncol = 6
        nrow = int(np.ceil(len(order) / ncol))
        figg, axg = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow),
                                 squeeze=False)
        H, W = raw.shape
        for k, idx in enumerate(order):
            x, y, r, pol, s = (d["x"][idx], d["y"][idx], d["r_est"][idx],
                               d["polarity"][idx], d["sym"][idx])
            hh = int(max(2 * r, 14))
            xi, yi = int(round(x)), int(round(y))
            a, b = max(0, xi - hh), max(0, yi - hh)
            cropg = raw[b:min(H, yi + hh), a:min(W, xi + hh)]
            ax = axg[k // ncol][k % ncol]
            ax.imshow(cropg, cmap="gray")
            cc = "lime" if pol >= 0 else "cyan"
            ax.plot((x - a) + r * np.cos(th), (y - b) + r * np.sin(th), "-",
                    color=cc, lw=1.3)
            ax.plot(x - a, y - b, "r+", ms=8)
            ax.set_title(f"sym={s:.2f} r={r:.0f}px", fontsize=8)
            ax.axis("off")
        for k in range(len(order), nrow * ncol):
            axg[k // ncol][k % ncol].axis("off")
        figg.suptitle(f"{args.run} f{args.frame}: 12 strongest detections "
                      f"(red+=center, circle=r_est)")
        pg = figstyle.save(figg, os.path.join(tdir, f"frst_f{args.frame}_gallery.png"))
        plt.close(figg)
        print(f"[detect-tune] wrote {pg}")
