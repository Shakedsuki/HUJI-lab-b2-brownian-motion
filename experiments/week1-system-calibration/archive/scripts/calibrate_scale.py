"""
calibrate_scale.py  (week1-system-calibration)
----------------------------------------------
Spatial calibration  um/px  for the 1632x1224 mode, from the ruler video
(a 0.1 mm = 100 um square grid). This is the CUBED lever: k_B is proportional
to (um/px)^3, so a 1% error here is 3% in k_B. We therefore measure the grid
period to sub-pixel precision and propagate an uncertainty.

Method (AUTO, default)
----------------------
1. Median several frames of the (static) ruler -> clean image, dust removed.
2. For each axis, COLLAPSE the perpendicular direction with a MEDIAN (not mean):
   a dark grid line is dark for almost every pixel along its length, so the
   median keeps it; the few rows/cols carrying the *crossing* lines are a
   minority and the median rejects them. -> one clean 1-D profile per axis.
3. Detect every dark line as a peak of the inverted profile, and locate each
   line CENTER to sub-pixel precision as the midpoint of its half-maximum
   crossings (robust for thick, flat-bottomed bars).
4. Assign each center an integer grid index (rounding to the median spacing, so
   a missing line does not corrupt the fit) and LINEAR-FIT center vs index.
   The slope is the period in px; its standard error gives the uncertainty.
5. um_per_px = pitch_um / period_px,  done for x and y, with an isotropy check.

    um_per_px = period_um / period_px        [um/px]
    d(k_B)/k_B = 3 * d(um_per_px)/um_per_px   (the cube)

Writes calibration/scale.json (+ keeps a human note) and calibration/scale_check.png
(profiles with detected centers) for you to eyeball the line count.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/calibrate_scale.py                       # ruler video, both axes
    python scripts/calibrate_scale.py --image calibration/ruler.png
    python scripts/calibrate_scale.py --dry-run             # report only, don't write scale.json
    # coarse fallback (two points read off ruler.png, 100 um apart on x):
    python scripts/calibrate_scale.py --manual --p1 365 --p2 1040 --dist-um 100 --axis x
"""

import argparse
import json
import os
import sys

import numpy as np

import _paths

DEFAULT_RULER = "measurent_0.1mm_focused.avi"


# --------------------------------------------------------------------------- IO
def _gray(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    return a.astype(np.float32)


def read_frames(path, n_sample=21):
    """Grayscale float32 frames sampled across the (static) ruler video."""
    try:
        import imageio.v3 as iio
        frames = []
        for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
            if i % 3 == 0:
                frames.append(_gray(fr))
            if len(frames) >= n_sample:
                break
        if frames:
            return frames
    except Exception as e:
        print(f"[imageio/pyav failed: {e}] falling back to OpenCV", file=sys.stderr)
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.linspace(0, max(total - 1, 0), n_sample).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(_gray(fr))
    cap.release()
    if not frames:
        sys.exit("opened the file but read no frames (codec issue).")
    return frames


def read_image(path):
    try:
        import imageio.v3 as iio
        return _gray(iio.imread(path))
    except Exception:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if a is None:
            sys.exit(f"could not read image {path}")
        return a.astype(np.float32)


# ----------------------------------------------------------------- line finding
def _find_peaks(d, height, distance):
    """Local maxima of d above `height`, no two within `distance` (keep tallest)."""
    try:
        from scipy.signal import find_peaks
        pk, _ = find_peaks(d, height=height, distance=distance)
        return list(pk)
    except Exception:
        cand = [i for i in range(1, len(d) - 1)
                if d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] >= height]
        cand.sort(key=lambda i: -d[i])
        kept = []
        for i in cand:
            if all(abs(i - j) >= distance for j in kept):
                kept.append(i)
        return sorted(kept)


def _half_max_center(d, pk):
    """Sub-pixel line center = midpoint of the half-maximum crossings of peak pk."""
    half = 0.5 * d[pk]
    i = pk
    while i > 0 and d[i] > half:
        i -= 1
    xl = i if d[i + 1] == d[i] else i + (half - d[i]) / (d[i + 1] - d[i])
    j = pk
    while j < len(d) - 1 and d[j] > half:
        j += 1
    xr = j if d[j] == d[j - 1] else (j - 1) + (half - d[j - 1]) / (d[j] - d[j - 1])
    on_edge = (i == 0) or (j == len(d) - 1)
    return 0.5 * (xl + xr), on_edge


def line_centers(profile, prom_frac, distance, edge_guard):
    """Return sub-pixel centers of dark lines in a 1-D profile (bright background)."""
    d = profile.max() - profile               # dark lines -> positive peaks
    height = prom_frac * (d.max() - d.min())
    peaks = _find_peaks(d, height=height, distance=distance)
    centers = []
    n = len(d)
    for pk in peaks:
        c, on_edge = _half_max_center(d, pk)
        if on_edge or c < edge_guard or c > n - 1 - edge_guard:
            continue                          # drop partial bars at the frame edge
        centers.append(c)
    return np.array(sorted(centers))


def fit_period(centers):
    """Linear-fit line center vs grid index. Returns dict with period_px + stderr."""
    n = len(centers)
    if n < 2:
        return None
    diffs = np.diff(centers)
    m0 = float(np.median(diffs))
    idx = np.round((centers - centers[0]) / m0).astype(int)   # robust to a skipped line
    if n == 2:
        period = float(centers[1] - centers[0]) / max(idx[1], 1)
        return dict(period_px=period, period_err=np.nan, rms_px=0.0,
                    n_lines=n, n_periods=int(idx[-1]))
    coef, cov = np.polyfit(idx, centers, 1, cov=True)
    period, b = float(coef[0]), float(coef[1])
    period_err = float(np.sqrt(cov[0, 0]))
    rms = float(np.std(centers - (period * idx + b)))
    return dict(period_px=period, period_err=period_err, rms_px=rms,
                n_lines=n, n_periods=int(idx[-1]))


def measure_axis(profile, pitch_um, prom_frac, distance, edge_guard):
    c = line_centers(profile, prom_frac, distance, edge_guard)
    fit = fit_period(c)
    if fit is None:
        return None, c
    fit["mpp"] = pitch_um / fit["period_px"]
    fit["mpp_err"] = (fit["mpp"] * fit["period_err"] / fit["period_px"]
                      if np.isfinite(fit["period_err"]) else np.nan)
    fit["centers"] = c
    return fit, c


# ----------------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description="Measure um/px from the ruler grid (sub-pixel).")
    p.add_argument("source", nargs="?", default=None,
                   help="ruler video or image; default = videos/" + DEFAULT_RULER)
    p.add_argument("--image", action="store_true", help="treat source as a still image")
    p.add_argument("--pitch-um", type=float, default=100.0, help="grid pitch (um); slide = 100")
    p.add_argument("--prom-frac", type=float, default=0.30, help="peak height as frac of profile range")
    p.add_argument("--min-sep", type=int, default=150, help="min px between distinct lines")
    p.add_argument("--edge-guard", type=int, default=8, help="ignore lines within N px of the edge")
    p.add_argument("--dry-run", action="store_true", help="report + plot, do NOT write scale.json")
    # manual fallback
    p.add_argument("--manual", action="store_true")
    p.add_argument("--p1", type=float); p.add_argument("--p2", type=float)
    p.add_argument("--dist-um", type=float, default=100.0)
    p.add_argument("--axis", choices=["x", "y"], default="x")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(_paths.CALIB_DIR, exist_ok=True)

    if args.manual:
        if args.p1 is None or args.p2 is None:
            sys.exit("--manual needs --p1 and --p2")
        period_px = abs(args.p2 - args.p1)
        mpp = args.dist_um / period_px
        print(f"[manual] {args.dist_um} um spans {period_px:.1f} px on {args.axis} -> "
              f"um_per_px = {mpp:.5f}")
        return

    # --- build a clean ruler image -----------------------------------------
    src = args.source or _paths.video(DEFAULT_RULER)
    if not os.path.exists(src):
        alt = os.path.join(_paths.CALIB_DIR, "ruler.png")
        if os.path.exists(alt):
            src, args.image = alt, True
        else:
            sys.exit(f"no ruler source found ({src} missing, no calibration/ruler.png)")
    if args.image:
        ruler = read_image(src); n_med = 1
        src_desc = f"image:{os.path.basename(src)}"
    else:
        frames = read_frames(src); ruler = np.median(np.stack(frames), axis=0)
        n_med = len(frames); src_desc = f"video:{os.path.basename(src)} (median of {n_med})"
    H, W = ruler.shape
    plt.imsave(os.path.join(_paths.CALIB_DIR, "ruler.png"), ruler, cmap="gray")

    # --- median-collapse profiles + per-axis period -------------------------
    prof_x = np.median(ruler, axis=0)   # vertical lines -> dips along x
    prof_y = np.median(ruler, axis=1)   # horizontal lines -> dips along y
    fx, cx = measure_axis(prof_x, args.pitch_um, args.prom_frac, args.min_sep, args.edge_guard)
    fy, cy = measure_axis(prof_y, args.pitch_um, args.prom_frac, args.min_sep, args.edge_guard)

    def show(tag, f):
        if not f:
            print(f"  [{tag}] <2 lines detected -- cannot measure on this axis"); return
        e = f"+/- {f['mpp_err']:.5f}" if np.isfinite(f["mpp_err"]) else "(no unc: only 2 lines)"
        print(f"  [{tag}] {f['n_lines']} lines / {f['n_periods']} period(s): "
              f"period = {f['period_px']:.2f} px (rms {f['rms_px']:.2f}px) -> "
              f"um_per_px = {f['mpp']:.5f} {e}")
    print(f"[ruler] {W}x{H}  {src_desc}")
    show("x", fx); show("y", fy)

    # --- combine axes -------------------------------------------------------
    cand = [f for f in (fx, fy) if f]
    if not cand:
        sys.exit("no axis yielded >=2 grid lines; check --prom-frac/--min-sep and scale_check.png")
    have_err = [f for f in cand if np.isfinite(f["mpp_err"]) and f["mpp_err"] > 0]
    if len(have_err) == len(cand) and len(cand) > 1:
        w = np.array([1 / f["mpp_err"] ** 2 for f in cand])
        mpp = float(np.sum(w * np.array([f["mpp"] for f in cand])) / w.sum())
        mpp_err = float(1 / np.sqrt(w.sum()))
    else:
        best = min(cand, key=lambda f: (np.inf if not np.isfinite(f["mpp_err"]) else f["mpp_err"]))
        mpp, mpp_err = best["mpp"], best["mpp_err"]
    aniso = (abs(fx["mpp"] - fy["mpp"]) / (0.5 * (fx["mpp"] + fy["mpp"])) * 100
             if (fx and fy) else float("nan"))
    print(f"\n[result] um_per_px = {mpp:.5f} +/- {mpp_err if np.isfinite(mpp_err) else float('nan'):.5f}"
          f"   (1 px = {mpp*1000:.1f} nm)")
    if np.isfinite(aniso):
        print(f"[result] x vs y isotropy: {aniso:.2f}%  ({'square pixels OK' if aniso<3 else 'CHECK: anisotropic!'})")
    if np.isfinite(mpp_err):
        print(f"[result] -> k_B uncertainty from calibration alone ~ {300*mpp_err/mpp:.1f}% (3x, the cube)")
    print(f"[compare] previous hand value 0.14390 -> change {(mpp/0.1439-1)*100:+.2f}%  "
          f"(k_B shifts ~{((mpp/0.1439)**3-1)*100:+.1f}%)")

    # --- diagnostic plot ----------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    for a, prof, f, name in ((ax[0], prof_x, fx, "x (vertical lines)"),
                             (ax[1], prof_y, fy, "y (horizontal lines)")):
        a.plot(prof, lw=0.8)
        if f is not None:
            for c in f["centers"]:
                a.axvline(c, color="r", lw=0.8, alpha=0.7)
            a.set_title(f"profile along {name}: {f['n_lines']} lines, "
                        f"period {f['period_px']:.2f}px, um/px {f['mpp']:.5f}")
        else:
            a.set_title(f"profile along {name}: <2 lines detected")
        a.set_ylabel("median intensity")
    ax[1].set_xlabel("pixel")
    fig.tight_layout()
    fig.savefig(os.path.join(_paths.CALIB_DIR, "scale_check.png"), dpi=120)
    print(f"[plot] wrote {os.path.join(_paths.CALIB_DIR, 'scale_check.png')}  (EYEBALL: one red line per grid line)")

    # --- write scale.json ---------------------------------------------------
    if args.dry_run:
        print("[dry-run] scale.json NOT written.")
        return

    def axis_block(f):
        if not f:
            return None
        return {"um_per_px": round(f["mpp"], 6),
                "um_per_px_unc": (round(f["mpp_err"], 6) if np.isfinite(f["mpp_err"]) else None),
                "period_px": round(f["period_px"], 3),
                "period_err_px": (round(f["period_err"], 3) if np.isfinite(f["period_err"]) else None),
                "n_lines": f["n_lines"], "n_periods": f["n_periods"], "rms_px": round(f["rms_px"], 3)}

    out = {
        "_note": ("Spatial calibration for the 1632x1224 mode (run2-run10). Sub-pixel "
                  "grid-line centers (half-max midpoints) on a median-collapsed ruler "
                  "profile; period from a linear fit of center-vs-index over all lines. "
                  "k_B scales as (um_per_px)^3, so this is the highest-leverage number."),
        "um_per_px": round(mpp, 6),
        "um_per_px_unc": (round(mpp_err, 6) if np.isfinite(mpp_err) else None),
        "pitch_um": args.pitch_um,
        "axes": {"x": axis_block(fx), "y": axis_block(fy)},
        "isotropy_pct": (round(aniso, 3) if np.isfinite(aniso) else None),
        "source": src_desc,
        "resolution": f"{W}x{H}",
        "applies_to": "run2..run10 (1632x1224 mode only; NOT run1/test at 2560x1920)",
        "objective": "40x",
        "method": "sub-pixel half-max line centers + linear period fit (both axes)",
    }
    with open(_paths.SCALE_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[write] {_paths.SCALE_JSON}  (um_per_px = {mpp:.5f})")


if __name__ == "__main__":
    main()
