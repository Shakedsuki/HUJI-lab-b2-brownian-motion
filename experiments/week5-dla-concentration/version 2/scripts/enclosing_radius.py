#!/usr/bin/env python3
"""Week 5 - DLA vs CuSO4 concentration: enclosing-circle radius R(t),
overlay videos, and growth kinetics.

Three depositions were recorded at CuSO4 concentrations 0.02 / 0.04 / 0.06
(same cell, ~12 V, central cathode wire), 1280x720 @ 59.94 fps, 11-16 min each.
Deliverables (per the instructor):

  1. enclosing-circle radius vs time for every sample,
  2. an overlay video showing that circle on the deposition footage,
  3. the fractal dimension for every concentration (see fractal_dimension.py;
     the mass-radius exponent M ~ Rg^D fitted here is the video cross-check).

Segmentation follows week 3 (temporal change vs a median-of-first-frames
reference + flat-field local-contrast hysteresis + wire mask + seed-connected
cluster gate) with two week-5 fixes:

  * the wire-hue band is tightened to GREEN (H 25-95); week 3 used H 18-95,
    which also caught the copper-brown central deposit and punched a hole in
    the mask;
  * the blue mm-grid paper visible at the left edge is excluded by blueness
    (deposit is neutral dark, B ~ G ~ R; grid lines have B >> R,G), so lighting
    drift over the paper can never masquerade as growth.

The mm-graph paper in frame gives a real px->mm calibration (autocorrelation
of the blue-line profile; ~34.6 px/mm, identical for the three runs since the
camera was not moved).  R(t) is reported in mm.

The enclosing circle is cv2.minEnclosingCircle of the seed-connected deposit.
Frames where the deposit touches the frame border are flagged (`edge`): from
the first contact on, the circle is a lower bound (drawn dashed / plotted
hollow).

Run:  python scripts/enclosing_radius.py [--runs run1,run2] [--sample-fps 2]
      [--no-video]  (video rendering is the slow part)
"""

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
VIDS = ROOT / "overlays"   # not "videos/": .gitignore drops mp4s there
VIDEO_DIR = Path(os.environ.get(
    "WEEK5_VIDEO_DIR",
    r"C:\dev\brownian-motion\experiments\week5-dla-concentration\raw-videos"))

FPS = 60000 / 1001
SAMPLE_FPS = 2.0            # measurement cadence [Hz]
REF_N = 12                  # frames median-ed into the static reference
OVERLAY_FPS = 30            # playback rate of the overlay video (x15 real time)

# segmentation constants (week 3 values unless noted)
FLAT_SIGMA = 101
HYST_HI = 0.15
HYST_LO = 0.06
CHANGE_THR = 12
S_WIRE = 100
H_WIRE_LO, H_WIRE_HI = 25, 95   # week-5 fix: green only, spare the copper hues
WIRE_DILATE = 18
BLUE_THR = 10               # B - (G+R)/2 above this = blue grid paper, not deposit
MIN_SIZE = 10
HUB_BRIDGE = 30
GAP_CLOSE = 4
CONNECT_NEAR = 24           # px; wire conducts connectivity only this close to deposit
STRONG_CORE = 0.25          # component max local-contrast score below this = shadow,
                            # not deposit (measured gap: dendrite >= 0.38, specks
                            # darkened by the moving wire shadow <= 0.21)
EDGE_PAD = 3                # px; deposit closer than this to the border = edge contact

RUNS = [
    dict(tag="run1_c0.02", file="run1_0.02con.mov", conc=0.02, label="run 1 - 0.02"),
    dict(tag="run2_c0.04", file="run 2 conc 0.04.mov", conc=0.04, label="run 2 - 0.04"),
    dict(tag="run3_c0.06", file="run3_0.06C.mov", conc=0.06, label="run 3 - 0.06"),
]

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


# ------------------------------------------------------------ calibration ---

def grid_pitch_px_per_mm(bgr, strip_w=130, max_lag=400):
    """px per 1 mm graph-paper square, from the autocorrelation of the blue
    horizontal-line profile in the left strip.  A linear fit through the first
    ~8 autocorrelation peaks (lag ~ k * pitch) beats reading off a single peak."""
    strip = bgr[:, :strip_w].astype(np.float32)
    b, g, r = strip[..., 0], strip[..., 1], strip[..., 2]
    prof = (b - (g + r) / 2).mean(axis=1)
    prof = prof - prof.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-9)
    # LOCAL maxima only (a windowed argmax can land on the window edge, still
    # on the flank of the previous peak -- the bug that halved the pitch)
    top = min(max_lag, len(ac) - 2)
    peaks = [l for l in range(6, top)
             if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1] and ac[l] > 0.05]
    base = [l for l in peaks if 20 <= l <= 50]
    if not base:
        raise RuntimeError("no grid-pitch peak in the 20-50 px window")
    pitch = float(base[0])
    lags, ks = [], []
    for k in range(1, 9):
        cand = [l for l in peaks if abs(l - k * pitch) < 0.3 * pitch]
        if not cand:
            break
        lag = min(cand, key=lambda l: abs(l - k * pitch))
        # parabolic sub-pixel refinement
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        lags.append(lag + float(np.clip(d, -1, 1)))
        ks.append(k)
        pitch = lags[-1] / k          # refine the harmonic prediction
    if len(ks) < 2:
        raise RuntimeError("grid autocorrelation found <2 harmonic peaks")
    coef = np.polyfit(ks, lags, 1)
    pitch = float(coef[0])
    resid = np.array(lags) - np.polyval(coef, ks)
    err = float(np.std(resid) / np.sqrt(len(ks))) if len(ks) > 2 else 0.5
    return pitch, max(err, 0.2)


# ------------------------------------------------------------ segmentation ---

def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def wire_mask(bgr, dilate=WIRE_DILATE):
    """The bright green cathode wire (the only moving coloured object).
    Hue band restricted to green so the copper-brown deposit is spared."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    w = ((s > S_WIRE) & (h > H_WIRE_LO) & (h < H_WIRE_HI) & (v > 60)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(w)
    out = np.zeros_like(w)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 400:
            out[lab == i] = 1
    return cv2.dilate(out, disk(dilate)) if dilate else out


def blue_grid_mask(bgr):
    """The blue mm-grid lines of the graph paper (B well above G,R)."""
    b, g, r = bgr[..., 0].astype(np.int16), bgr[..., 1].astype(np.int16), bgr[..., 2].astype(np.int16)
    return (b - (g + r) // 2 > BLUE_THR).astype(np.uint8)


def _hysteresis(score, hi, lo):
    strong = (score > hi).astype(np.uint8)
    weak = (score > lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak)
    keep = set(np.unique(lab[strong > 0])) - {0}
    return np.isin(lab, list(keep)).astype(np.uint8) if keep else np.zeros_like(weak)


def _flatfield_bg(gray, sigma=FLAT_SIGMA, scale=0.25):
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sb = cv2.GaussianBlur(small, (0, 0), sigma * scale)
    return cv2.resize(sb, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def deposit_mask(bgr, ref_gray, hi=HYST_HI, lo=HYST_LO):
    """New dark deposit = flat-field-dark AND darkened-since-start AND
    not-wire AND not-blue-grid, despeckled.

    Components must also contain a STRONG dark core (score >= STRONG_CORE):
    the wire's moving shadow darkens patches of paper enough to pass the
    change+hysteresis test, but only weakly (score <= 0.21), while any real
    deposit fragment has an essentially black core (>= 0.38).  Without this
    gate the shadow specks chain into the cluster along the wire and balloon
    the enclosing circle."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = _flatfield_bg(gray)
    score = 1.0 - gray / (bg + 1e-6)
    m = _hysteresis(score, hi, lo)
    offset = np.median(gray) - np.median(ref_gray)
    changed = (ref_gray - (gray - offset)) > CHANGE_THR
    m = (m & changed).astype(np.uint8)
    m[wire_mask(bgr) > 0] = 0
    m[blue_grid_mask(bgr) > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_SIZE:
            continue
        sel = lab == i
        if score[sel].max() >= STRONG_CORE:
            out[sel] = 1
    return out


def cluster_gate(mask, seed, connector=None):
    """Keep only the deposit connected to the cathode-tip seed.

    ``connector`` (the DILATED wire mask, i.e. exactly the region deposit_mask
    excludes) counts for connectivity but not for mass: the deposit physically
    continues underneath the wire, so branches the wire exclusion splits off --
    and the seed disk itself, which the exclusion ring-fences in late run-3
    frames, zeroing the whole gated mask -- stay connected through it.  The
    undilated wire body is NOT enough: the exclusion leaves an 18 px moat
    between the wire and the nearest deposit pixel.

    The connector is restricted to where it is ADJACENT to deposit (within
    CONNECT_NEAR px): a long bare stretch of wire crossing the graph paper must
    not conduct connectivity, or specks darkened by the wire's own moving
    shadow far from the aggregate get dragged into the cluster and balloon the
    enclosing circle."""
    if mask.sum() == 0:
        return mask
    bridged = mask.copy()
    if connector is not None:
        near = cv2.dilate(mask, disk(CONNECT_NEAR))
        bridged |= (connector & near)
    cv2.circle(bridged, seed, HUB_BRIDGE, 1, -1)
    if GAP_CLOSE:
        bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, disk(GAP_CLOSE))
    n, lab = cv2.connectedComponents(bridged)
    sx, sy = seed
    seedlab = lab[sy, sx]
    if seedlab == 0:
        return np.zeros_like(mask)
    return (mask * (lab == seedlab)).astype(np.uint8)


# --------------------------------------------------------- video iteration ---

def extract_frames(path, tmp, sample_fps):
    refdir, samp = tmp / "ref", tmp / "f"
    refdir.mkdir(); samp.mkdir()
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-frames:v", str(REF_N), str(refdir / "r_%03d.png")], check=True)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-vf", f"fps={sample_fps}", "-q:v", "2", str(samp / "f_%05d.jpg")],
                   check=True)
    refs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for p in sorted(refdir.glob("r_*.png"))]
    ref = np.median(np.stack(refs), axis=0)
    frames = [(k / sample_fps, p) for k, p in enumerate(sorted(samp.glob("f_*.jpg")))]
    return ref, frames


def find_seed(frames, ref):
    """Cathode tip = persistent early-deposit pixel closest to the wire."""
    acc = np.zeros(ref.shape, np.float32)
    wu = np.zeros(ref.shape, np.uint8)
    early = frames[:max(int(0.30 * len(frames)), 3)]
    step = max(len(early) // 40, 1)          # ~40 frames are plenty
    for _, p in early[::step]:
        f = cv2.imread(str(p))
        acc += deposit_mask(f, ref)
        wu |= wire_mask(f)
    accb = cv2.GaussianBlur(acc, (0, 0), 6)
    if accb.max() == 0:
        return (ref.shape[1] // 2, ref.shape[0] // 2)
    persistent = accb > 0.5 * accb.max()
    if wu.sum() > 0:
        wdist = cv2.distanceTransform((wu == 0).astype(np.uint8), cv2.DIST_L2, 5)
        cand = np.where(persistent, wdist, np.inf)
        sy, sx = np.unravel_index(np.argmin(cand), cand.shape)
    else:
        sy, sx = np.unravel_index(np.argmax(accb), accb.shape)
    return (int(sx), int(sy))


def enclosing_circle(tree):
    """Minimal enclosing circle of the deposit, via the convex hull (exact and
    fast: the MEC is determined by hull points only)."""
    ys, xs = np.nonzero(tree)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    return float(cx), float(cy), float(r)


def draw_overlay(frame, tree, seed, circ, t_s, px_per_mm, edge):
    """Instructor deliverable 2: the enclosing circle drawn on the footage."""
    vis = frame.copy()
    # deposit tint (subtle: blend green into deposit pixels)
    sel = tree > 0
    vis[sel] = (0.55 * vis[sel] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    cx, cy, r = circ
    colour = (0, 0, 255) if not edge else (0, 160, 255)
    cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), colour, 2,
               lineType=cv2.LINE_AA)
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, colour, -1, lineType=cv2.LINE_AA)
    cv2.drawMarker(vis, seed, (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
    label = f"t = {t_s:6.1f} s   R = {r / px_per_mm:5.2f} mm"
    if edge:
        label += "  (edge: lower bound)"
    cv2.putText(vis, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(vis, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                cv2.LINE_AA)
    # 5 mm scale bar, bottom right
    L = int(round(5 * px_per_mm))
    x1, y1 = vis.shape[1] - 40, vis.shape[0] - 30
    cv2.line(vis, (x1 - L, y1), (x1, y1), (0, 0, 0), 5)
    cv2.line(vis, (x1 - L, y1), (x1, y1), (255, 255, 255), 2)
    cv2.putText(vis, "5 mm", (x1 - L, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, "5 mm", (x1 - L, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def measure(run, frames, ref, seed, px_per_mm, video_out=None):
    """Per sampled frame: mass, gyration radius, seed radii, enclosing circle,
    edge-contact flag.  Optionally streams the overlay video to ffmpeg.

    Cluster membership uses MEMORY on top of the per-frame seed connectivity:
    the deposit is immobile and permanent, so any current component overlapping
    the past cluster belongs to it.  This keeps branches whose root crosses the
    static wire-shadow band (dark since frame 1, so the change mask never sees
    the root) from flickering out when the diffuse halo that used to connect
    them fades.  Mass still counts only currently-dark pixels."""
    rows = []
    sx, sy = seed
    H, W = ref.shape
    memory = np.zeros_like(ref, dtype=np.uint8)
    enc = None
    if video_out is not None:
        enc = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(OVERLAY_FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
             "-pix_fmt", "yuv420p", str(video_out)],
            stdin=subprocess.PIPE)
    try:
        for t, p in frames:
            f = cv2.imread(str(p))
            base = deposit_mask(f, ref)
            tree = cluster_gate(base, seed, connector=wire_mask(f))
            if memory.any() and base.any():
                n, lab = cv2.connectedComponents(base)
                memd = cv2.dilate(memory, disk(8))
                hit = np.unique(lab[(memd > 0) & (lab > 0)])
                if len(hit):
                    tree = (tree | (np.isin(lab, hit) & (base > 0))).astype(np.uint8)
            memory |= tree
            ys, xs = np.nonzero(tree)
            if len(xs) < 5:
                rows.append((t, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                if enc is not None:
                    enc.stdin.write(f.tobytes())
                continue
            d = np.hypot(xs - sx, ys - sy)
            cx_m, cy_m = xs.mean(), ys.mean()
            rg = float(np.sqrt(((xs - cx_m) ** 2 + (ys - cy_m) ** 2).mean()))
            ccx, ccy, cr = enclosing_circle(tree)
            edge = int((xs.min() <= EDGE_PAD) or (ys.min() <= EDGE_PAD) or
                       (xs.max() >= W - 1 - EDGE_PAD) or (ys.max() >= H - 1 - EDGE_PAD))
            rows.append((t, int(tree.sum()), rg, float(np.percentile(d, 95)),
                         float(d.max()), ccx, ccy, cr, edge,
                         int(cv2.connectedComponents(tree)[0] - 1)))
            if enc is not None:
                vis = draw_overlay(f, tree, seed, (ccx, ccy, cr), t, px_per_mm, edge)
                enc.stdin.write(vis.tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
    return np.array(rows, dtype=float)


# ----------------------------------------------------------------- fitting ---

def fit_loglog(x, y):
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    dof = max(len(lx) - 2, 1)
    rv = (res[0] / dof) if len(res) else 0.0
    cov = rv * np.linalg.inv(A.T @ A)
    return coef[0], float(np.sqrt(cov[0, 0])), coef[1]


def hampel_inliers(y, k=5, nsig=3.5):
    y = np.asarray(y, float)
    inl = np.ones(len(y), bool)
    for i in range(len(y)):
        lo, hi = max(0, i - k), min(len(y), i + k + 1)
        seg = y[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-9
        if abs(y[i] - med) > nsig * 1.4826 * mad:
            inl[i] = False
    return inl


def nucleation_time(t, M, frac=0.02):
    Mref = np.percentile(M[M > 0], 98) if np.any(M > 0) else 0.0
    hit = np.where(M >= frac * Mref)[0]
    return float(t[hit[0]]) if len(hit) else 0.0


def growth_window(M, R, edge, lo_frac=0.08, hi_frac=0.75):
    """Self-similar growth regime, restricted to frames BEFORE the deposit
    touches the border (after that M and R are both censored)."""
    ok = edge == 0
    Mref = np.percentile(M[(M > 0) & ok], 98) if np.any((M > 0) & ok) else 0.0
    w = (M >= lo_frac * Mref) & (M <= hi_frac * Mref) & (R > 0) & (M > 0) & ok
    w &= hampel_inliers(R) & hampel_inliers(M)
    return w


# ---------------------------------------------------------------- pipeline ---

def process(run, sample_fps, render_video):
    path = VIDEO_DIR / run["file"]
    tmp = Path(tempfile.mkdtemp(prefix=f"w5_{run['tag']}_"))
    try:
        ref, frames = extract_frames(path, tmp, sample_fps)
        first = cv2.imread(str(frames[0][1]))
        px_per_mm, dpitch = grid_pitch_px_per_mm(first)
        seed = find_seed(frames, ref)
        video_out = (VIDS / f"overlay_{run['tag']}.mp4") if render_video else None
        if video_out is not None:
            VIDS.mkdir(exist_ok=True)
        rows = measure(run, frames, ref, seed, px_per_mm, video_out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    DATA.mkdir(exist_ok=True)
    with open(DATA / f"radius_{run['tag']}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "M_px", "Rg_px", "R95seed_px", "Rmaxseed_px",
                     "circ_cx_px", "circ_cy_px", "circ_R_px", "edge", "n_comp"])
        wr.writerows(rows)
        fh.write(f"# px_per_mm = {px_per_mm:.3f} +/- {dpitch:.3f}\n")
        fh.write(f"# seed = {seed}\n")

    t, M, Rg, R95, Rmax, ccx, ccy, cR, edge, ncomp = rows.T
    res = dict(run=run, t=t, M=M, Rg=Rg, Rc=cR, edge=edge, seed=seed,
               px_per_mm=px_per_mm, dpitch=dpitch)
    w = growth_window(M, Rg, edge)
    t0 = nucleation_time(t, M)
    res["t0"], res["win"] = t0, w
    tau = t - t0
    m = w & (tau > 0)
    if m.sum() >= 8:
        res["alpha"], res["dalpha"], _ = fit_loglog(tau[m], M[m])
        res["beta"], res["dbeta"], _ = fit_loglog(tau[m], cR[m])
        res["betaRg"], res["dbetaRg"], _ = fit_loglog(tau[m], Rg[m])
        res["D"], res["dD"], _ = fit_loglog(Rg[m], M[m])
        # window systematic
        Ds, bs = [], []
        for lo, hi in [(0.08, 0.75), (0.05, 0.85), (0.12, 0.65), (0.10, 0.70)]:
            ww = growth_window(M, Rg, edge, lo, hi) & (tau > 0)
            if ww.sum() >= 8:
                Ds.append(fit_loglog(Rg[ww], M[ww])[0])
                bs.append(fit_loglog(tau[ww], cR[ww])[0])
        half = lambda a: (max(a) - min(a)) / 2 if len(a) > 1 else 0.0
        res["D_sys"], res["beta_sys"] = half(Ds), half(bs)
    return res


# ----------------------------------------------------------------- figures ---

def radius_figure(res):
    """Deliverable 1: enclosing-circle radius vs time for one sample."""
    run, ppm = res["run"], res["px_per_mm"]
    t, Rc, edge = res["t"], res["Rc"], res["edge"]
    has = res["M"] > 0
    ok, ed = (edge == 0) & has, (edge == 1) & has
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t[ok], Rc[ok] / ppm, ".", ms=3, color="C0",
            label="enclosing-circle radius")
    if ed.any():
        ax.plot(t[ed], Rc[ed] / ppm, ".", ms=3, color="C1", alpha=0.6,
                label="deposit touches frame edge (lower bound)")
    if "beta" in res:
        tau = np.linspace(max(res["t0"] + 1, 1), t[res["win"]].max(), 200)
        m = res["win"] & (t - res["t0"] > 0)
        A = np.exp(np.median(np.log(Rc[m]) - res["beta"] * np.log(t[m] - res["t0"])))
        ax.plot(tau, A * (tau - res["t0"]) ** res["beta"] / ppm, "k-", lw=1.2,
                label=(f"fit  R ~ (t-t0)^beta,  beta = {res['beta']:.2f} "
                       f"+/- {np.hypot(res['dbeta'], res.get('beta_sys', 0)):.2f}"))
    ax.set_xlabel("t [s]")
    ax.set_ylabel("R [mm]")
    ax.set_title(f"{run['label']} % CuSO4 - enclosing-circle radius vs time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / f"R_vs_t_{run['tag']}.png"
    FIGS.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def summary_figure(results):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, r in enumerate(results):
        ppm = r["px_per_mm"]
        t, Rc, edge, w = r["t"], r["Rc"], r["edge"], r["win"]
        lab = f"{r['run']['conc']:.2f}"
        has = r["M"] > 0
        ok, ed = (edge == 0) & has, (edge != 0) & has
        ax[0].plot(t[ok], Rc[ok] / ppm, ".", ms=2.5, color=f"C{i}", label=lab)
        ax[0].plot(t[ed], Rc[ed] / ppm, ".", ms=2.5, color=f"C{i}", alpha=0.25)
        if "D" in r:
            m = w
            ax[1].loglog(r["Rg"][m] / ppm, r["M"][m] / ppm ** 2, "o", ms=2.5,
                         color=f"C{i}", label=f"{lab}:  D = {r['D']:.2f}")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("enclosing-circle R [mm]")
    ax[0].set_title("enclosing-circle radius vs time (faded = edge-censored)")
    ax[0].legend(title="CuSO4 conc.", fontsize=9); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("Rg [mm]"); ax[1].set_ylabel("M [mm^2]")
    ax[1].set_title("mass-radius  M ~ Rg^D")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    out = FIGS / "R_vs_t_all.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="", help="comma-separated tags (default all)")
    ap.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    want = set(s.strip() for s in args.runs.split(",") if s.strip())
    runs = [r for r in RUNS if not want or r["tag"] in want]

    results = []
    for run in runs:
        print(f"[{run['tag']}] {run['file']} ...", flush=True)
        r = process(run, args.sample_fps, not args.no_video)
        results.append(r)
        Rnz = r["Rc"][r["M"] > 0]
        msg = (f"  seed={r['seed']}  px/mm={r['px_per_mm']:.2f}  t0={r['t0']:.0f}s  "
               f"final R={(Rnz[-1] if len(Rnz) else 0) / r['px_per_mm']:.1f} mm")
        if "beta" in r:
            msg += (f"  beta(Rc~t)={r['beta']:.2f}+/-{np.hypot(r['dbeta'], r.get('beta_sys', 0)):.2f}"
                    f"  alpha(M~t)={r['alpha']:.2f}"
                    f"  D(M~Rg)={r['D']:.2f}+/-{np.hypot(r['dD'], r.get('D_sys', 0)):.2f}")
        print(msg, flush=True)
        radius_figure(r)
    if len(results) > 1:
        summary_figure(results)
    print(f"figures -> {FIGS}", flush=True)


if __name__ == "__main__":
    main()
