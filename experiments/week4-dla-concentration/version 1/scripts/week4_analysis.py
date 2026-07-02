#!/usr/bin/env python3
"""Week 4 — DLA growth vs CuSO4 concentration (0.15 / 0.30 / 0.45 / 0.56).

Same pipeline as week 5 (see week5-dla-concentration/version 1), two changes:
four runs, and the wire-mask hue band starts at 25 (not 18) so the
copper-brown deposit of the high-concentration runs is not masked out.

Deliverables (per Shlomo's request):
  1. Bounding-circle radius R(t) for every sample        -> data/radius_<tag>.csv,
                                                            figures/radius_vs_time_all.png
  2. Overlay of the bounding circle on the video         -> overlays/overlay_<tag>.mp4
  3. Fractal dimension per concentration                 -> figures/fractalD_<tag>.png

Method: reuses the week-3 pipeline (growth_kinetics.py) — segmentation is a
temporal "darkened since start" mask AND flat-field hysteresis threshold,
minus the HSV-segmented cathode wire, gated to the component connected to the
seed (cathode tip).  New here:
  * R(t) is the MINIMUM ENCLOSING CIRCLE of the seed-connected deposit
    (cv2.minEnclosingCircle) — the literal "מעגל חוסם".
  * mm calibration from the millimetric graph paper visible at the left edge
    (autocorrelation of the blue-grid intensity profile -> px per mm).
  * frames where the deposit touches the frame border are flagged "clipped"
    (the enclosing circle is then a lower bound) and excluded from fits.
  * fractal dimension per concentration by box counting on the final mask
    (+ mass-radius M ~ Rg^D from the growth as a cross-check).

Frames must be pre-extracted (ffmpeg, fps=1) into FRAMES_DIR/<tag>/{ref,f}.
Run:  python3 week5_analysis.py [--runs run1,run2,run3] [--stage measure|figs|overlay]
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # .../analysis
DATA = ROOT / "data"
FIGS = ROOT / "figures"
OVER = ROOT / "overlays"
FRAMES_DIR = Path(os.environ.get("W4_FRAMES", "/tmp/w4"))
TMP = Path(os.environ.get("W4_TMP", "/tmp/w4"))

SAMPLE_DT = 1.0                         # s between extracted frames (fps=1)

# segmentation constants — carried over from week-3 growth_kinetics.py
FLAT_SIGMA = 101
HYST_HI = 0.15
HYST_LO = 0.06
CHANGE_THR = 12
S_WIRE = 100
WIRE_DILATE = 18
MIN_SIZE = 10
HUB_BRIDGE = 30
GAP_CLOSE = 4
EDGE_PX = 3                             # deposit within this of border => clipped

RUNS = [
    dict(tag="run1", conc=0.56, label="run 1 — 0.56"),
    dict(tag="run2", conc=0.45, label="run 2 — 0.45"),
    dict(tag="run3", conc=0.30, label="run 3 — 0.30"),
    dict(tag="run4", conc=0.15, label="run 4 — 0.15"),
]


# ---------------------------------------------------------------- segmentation

def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def wire_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    w = ((s > S_WIRE) & (h > 25) & (h < 95) & (v > 60)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(w)
    out = np.zeros_like(w)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 400:
            out[lab == i] = 1
    return cv2.dilate(out, disk(WIRE_DILATE))


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


def register(bgr, ref_gray, scale=0.5):
    """Align the frame to the reference (phase correlation on downscaled
    grays).  A camera bump shifts the whole frame a few px, which otherwise
    paints false 'darkened' edges along every static high-contrast line
    (worst: the graph-paper grid, which then connects to the cluster through
    the wire shadow and blows up the bounding circle)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    r = cv2.resize(ref_gray.astype(np.float32), None, fx=scale, fy=scale,
                   interpolation=cv2.INTER_AREA)
    (dx, dy), _ = cv2.phaseCorrelate(r, g)
    dx, dy = dx / scale, dy / scale
    if abs(dx) < 0.4 and abs(dy) < 0.4:        # sub-pixel jitter: don't resample
        return bgr, float(dx), float(dy)
    Mw = np.float32([[1, 0, -dx], [0, 1, -dy]])
    reg = cv2.warpAffine(bgr, Mw, (bgr.shape[1], bgr.shape[0]),
                         borderMode=cv2.BORDER_REPLICATE)
    return reg, float(dx), float(dy)


def deposit_mask(bgr, ref_gray, hi=HYST_HI, lo=HYST_LO):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = _flatfield_bg(gray)
    score = 1.0 - gray / (bg + 1e-6)
    m = _hysteresis(score, hi, lo)
    offset = np.median(gray) - np.median(ref_gray)
    changed = (ref_gray - (gray - offset)) > CHANGE_THR
    m = (m & changed).astype(np.uint8)
    m[wire_mask(bgr) > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_SIZE:
            out[lab == i] = 1
    return out


def cluster_gate(mask, seed):
    if mask.sum() == 0:
        return mask
    bridged = mask.copy()
    cv2.circle(bridged, seed, HUB_BRIDGE, 1, -1)
    if GAP_CLOSE:
        bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, disk(GAP_CLOSE))
    n, lab = cv2.connectedComponents(bridged)
    sx, sy = seed
    seedlab = lab[sy, sx]
    if seedlab == 0:
        return np.zeros_like(mask)
    return (mask * (lab == seedlab)).astype(np.uint8)


# ------------------------------------------------------------- mm calibration

def mm_per_px(ref_bgr, strip_w=None, diag_path=None):
    """Pitch of the millimetric graph paper at the left edge.

    The grid is light blue on white: blue lines are dark in the RED channel.
    Column-averaged red profile over the grid strip -> autocorrelation ->
    first peak = pitch in px = 1 mm (paper is 1 mm ruled)."""
    red = ref_bgr[..., 2].astype(np.float32)
    blue = ref_bgr[..., 0].astype(np.float32)
    # white-balance the channels first (run 2 has a strong warm cast that
    # otherwise kills the blue-line contrast)
    blue = blue * (128.0 / (np.median(blue) + 1e-6))
    red = red * (128.0 / (np.median(red) + 1e-6))
    blueness = blue - red                       # grid lines: blue >> red
    col_score = (blueness > 8).mean(axis=0)
    grid_cols = np.where(col_score > 0.15)[0]
    if len(grid_cols) < 20:
        return None, None
    x1 = grid_cols.max() + 1
    strip = blueness[:, :x1]
    prof = strip.mean(axis=1)                   # horizontal lines -> y-profile
    # detrend: remove the low-frequency envelope (the wire crossing the strip
    # in week-4 run 2 adds a huge trough that buries the grid periodicity)
    trend = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 25).ravel()
    prof = prof - trend
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    ac /= ac[0] + 1e-9
    # fundamental = lag maximizing ac[k] + ac[2k] (harmonic validation kills
    # subharmonics and noise peaks; runs 2-3 of week 4 need this)
    lo, hi = 8, 80
    cand = [(ac[k] + (ac[2 * k] if 2 * k < len(ac) else -1), k)
            for k in range(lo, min(hi, len(ac) // 2))]
    k = max(cand)[1]
    # refine: parabolic interpolation
    if 1 <= k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom < -1e-12:                     # genuine maximum only
            k = k + 0.5 * (y0 - y2) / denom
    if diag_path:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        axs[0].imshow(cv2.cvtColor(ref_bgr[:, :x1 + 40], cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"grid strip (x < {x1})"); axs[0].axis("off")
        axs[1].plot(ac[:120]); axs[1].axvline(k, color="r", ls="--")
        axs[1].set_title(f"autocorr of y-profile: pitch = {k:.2f} px = 1 mm")
        fig.tight_layout(); fig.savefig(diag_path, dpi=110); plt.close(fig)
    return 1.0 / k, k                           # mm/px, px per mm


# ---------------------------------------------------------------- measurement

def load_ref(tag):
    refs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for p in sorted((FRAMES_DIR / tag / "ref").glob("r_*.png"))]
    ref_bgr = cv2.imread(str(sorted((FRAMES_DIR / tag / "ref").glob("r_*.png"))[0]))
    return np.median(np.stack(refs), axis=0), ref_bgr


def frames_of(tag):
    return [(k * SAMPLE_DT, p) for k, p in
            enumerate(sorted((FRAMES_DIR / tag / "f").glob("f_*.jpg")))]


def find_seed(frames, ref):
    acc = np.zeros(ref.shape, np.float32)
    wu = np.zeros(ref.shape, np.uint8)
    early = frames[:max(int(0.30 * len(frames)), 3)]
    for _, p in early[::6]:
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


def seed_run(run):
    """Once per run: growth seed + mm calibration -> data/meta_<tag>.json."""
    tag = run["tag"]
    ref, ref_bgr = load_ref(tag)
    frames = frames_of(tag)
    seed = find_seed(frames, ref)
    FIGS.mkdir(parents=True, exist_ok=True)
    mmpx, pitch = mm_per_px(ref_bgr, diag_path=FIGS / f"calibration_{tag}.png")
    DATA.mkdir(parents=True, exist_ok=True)
    meta = dict(tag=tag, conc=run["conc"], seed=[int(seed[0]), int(seed[1])],
                mm_per_px=float(mmpx), px_per_mm=float(pitch),
                n_frames=len(frames))
    with open(DATA / f"meta_{tag}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[{tag}] seed={seed} mm/px={mmpx:.5f} (pitch {pitch:.2f} px/mm) "
          f"frames={len(frames)}", flush=True)
    return meta


def measure_run(run, k0=0, k1=None):
    """Measure frames [k0, k1) -> part CSV in TMP/parts.

    Sequential with checkpointed state (chunks must run in ascending k):
      * each frame is REGISTERED to the reference (kills camera-bump
        artefacts);
      * TEMPORAL PERSISTENCE: a pixel is deposit only if dark now AND in >=1
        of the two previous frames (kills 1-frame transients: hands, glints);
      * MONOTONE ACCUMULATION: once a pixel is in the seed-connected cluster
        it stays (a deposit cannot shrink) -- branch dropouts behind the wire
        no longer dent R(t), and connectivity is evaluated on the union of
        the accumulated cluster and the current mask."""
    tag = run["tag"]
    ref, _ = load_ref(tag)
    frames = frames_of(tag)
    meta = json.load(open(DATA / f"meta_{tag}.json"))
    seed = tuple(meta["seed"])
    H, W = ref.shape
    k1 = len(frames) if k1 is None else min(k1, len(frames))

    hulls = {}
    if k0 == 0:
        acc = np.zeros((H, W), np.uint8)
        raw1 = raw2 = None
    else:
        st = np.load(TMP / f"state_{tag}_{k0:05d}.npz")
        acc, raw1, raw2 = (st["acc"].astype(np.uint8),
                           st["raw1"].astype(np.uint8),
                           st["raw2"].astype(np.uint8))

    rows = []
    for k in range(k0, k1):
        t, p = frames[k]
        f = cv2.imread(str(p))
        f, dx, dy = register(f, ref)
        shift = max(abs(dx), abs(dy))
        raw = deposit_mask(f, ref)
        # registration smears a border band ~|shift| wide (BORDER_REPLICATE
        # + non-overlap); exclude it from the mask
        b = int(min(140, np.ceil(shift) + 8))
        if b > 8:
            raw[:b, :] = 0; raw[-b:, :] = 0; raw[:, :b] = 0; raw[:, -b:] = 0
        if raw1 is None:
            persist = raw
        elif raw2 is None:
            persist = raw & raw1
        else:
            persist = raw & (raw1 | raw2)
        raw2, raw1 = raw1, raw
        # camera re-framed (week-4 runs were stopped/adjusted mid-run) or an
        # anomalous mask burst: freeze the cluster instead of absorbing junk
        new_px = int((persist & (1 - acc)).sum())
        moved = int(shift > 60 or new_px > 30000)
        if not moved:
            acc = cluster_gate((acc | persist).astype(np.uint8), seed)
        ys, xs = np.nonzero(acc)
        if len(xs) < 5:
            rows.append((k, t, 0, 0, 0, 0, 0, 0, 0, int(raw.sum()), dx, dy, moved))
            continue
        pts = np.column_stack([xs, ys]).astype(np.int32)
        hull = cv2.convexHull(pts)               # minEnclosingCircle is O(n) but
        (cx, cy), R = cv2.minEnclosingCircle(    # slow on 10^5 pts; hull ~ 10^2
            hull.reshape(-1, 2).astype(np.float32))
        hulls[k] = hull.reshape(-1, 2).tolist()  # farthest-point queries for ANY
                                                 # fixed centre need only the hull
        rg = float(np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean()))
        d = np.hypot(xs - seed[0], ys - seed[1])
        clipped = int((xs.min() <= EDGE_PX) or (ys.min() <= EDGE_PX) or
                      (xs.max() >= W - 1 - EDGE_PX) or (ys.max() >= H - 1 - EDGE_PX))
        rows.append((k, t, int(acc.sum()), float(R), float(cx), float(cy),
                     rg, float(np.percentile(d, 95)), clipped, int(raw.sum()),
                     dx, dy, moved))
    np.savez_compressed(
        TMP / f"state_{tag}_{k1:05d}.npz", acc=acc,
        raw1=(raw1 if raw1 is not None else np.zeros((H, W), np.uint8)),
        raw2=(raw2 if raw2 is not None else np.zeros((H, W), np.uint8)))
    parts = TMP / "parts"
    parts.mkdir(exist_ok=True)
    with open(parts / f"{tag}_{k0:05d}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with open(parts / f"hull_{tag}_{k0:05d}.json", "w") as fh:
        json.dump(hulls, fh)
    print(f"[{tag}] measured {k0}..{k1}", flush=True)


def merge_run(run):
    """Concatenate part CSVs -> data/radius_<tag>.csv + circles json.

    The published circle uses the FIXED per-run centre (meta "center_fixed",
    the enclosing-circle centre of the final deposit): R_fix(k) = distance
    from that centre to the farthest hull vertex of frame k.  The centre
    never moves during a run (it marks the deposition site)."""
    tag = run["tag"]
    meta = json.load(open(DATA / f"meta_{tag}.json"))
    Cx, Cy = meta.get("center_fixed", meta["seed"])
    hulls = {}
    for p in sorted((TMP / "parts").glob(f"hull_{tag}_*.json")):
        hulls.update(json.load(open(p)))
    parts = sorted((TMP / "parts").glob(f"{tag}_[0-9]*.csv"))
    allrows = {}
    for p in parts:
        with open(p) as fh:
            for row in csv.reader(fh):
                if row:
                    allrows[int(row[0])] = [float(v) for v in row[1:]]
    for k, v in allrows.items():
        h = hulls.get(str(k))
        if h and v[2] > 0:
            pts = np.asarray(h, float)
            rfix = float(np.hypot(pts[:, 0] - Cx, pts[:, 1] - Cy).max())
        else:
            rfix = 0.0
        v.append(rfix)
    circ = {k: (Cx, Cy, v[-1]) for k, v in allrows.items() if v[-1] > 0}
    with open(DATA / f"radius_{tag}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "M_px", "Renc_px", "cx_px", "cy_px",
                     "Rg_px", "R95_px", "clipped", "npix", "dx_px", "dy_px",
                     "moved", "Rfix_px"])
        for k in sorted(allrows):
            wr.writerow(allrows[k])
    with open(TMP / f"circles_{tag}.json", "w") as fh:
        json.dump(circ, fh)
    print(f"[{tag}] merged {len(allrows)} rows from {len(parts)} parts", flush=True)


# ------------------------------------------------------------------- overlays

def overlay_run(run, k0=0, k1=None, speed=30):
    """Timelapse (1 fps sampled, played at 30 fps => x30) with the bounding
    circle drawn per frame (chunk-safe: draws frames [k0, k1))."""
    tag = run["tag"]
    frames = frames_of(tag)
    circ = json.load(open(TMP / f"circles_{tag}.json"))
    meta = json.load(open(DATA / f"meta_{tag}.json"))
    rows = np.genfromtxt(DATA / f"radius_{tag}.csv", delimiter=",", names=True)
    seed = tuple(meta["seed"])
    outdir = TMP / f"ov_{tag}"
    outdir.mkdir(exist_ok=True)
    mm = meta["mm_per_px"]
    k1 = len(frames) if k1 is None else min(k1, len(frames))
    for k in range(k0, k1):
        t, p = frames[k]
        f = cv2.imread(str(p))
        key = str(k)
        if key in circ:
            cx, cy, R = circ[key]
            # circles live in reference coordinates; shift back onto this
            # (unregistered) frame
            cx += float(rows["dx_px"][k])
            cy += float(rows["dy_px"][k])
            clipped = bool(rows["clipped"][k])
            col = (0, 200, 255) if not clipped else (0, 120, 255)
            cv2.circle(f, (int(round(cx)), int(round(cy))), int(round(R)), col, 2)
            cv2.circle(f, (int(round(cx)), int(round(cy))), 3, col, -1)
            txt = f"t = {t:5.0f} s   R = {R * mm:5.2f} mm" + \
                  ("  [clipped]" if clipped else "")
        else:
            txt = f"t = {t:5.0f} s"
        cv2.circle(f, seed, 4, (255, 80, 80), -1)
        cv2.putText(f, txt, (12, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(f, txt, (12, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        lbl = run["label"].replace("—", "-")   # cv2 putText is ASCII-only
        cv2.putText(f, f"{lbl}   (x{speed} timelapse)", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(f, f"{lbl}   (x{speed} timelapse)", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(outdir / f"o_{k:05d}.jpg"), f,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[{tag}] overlay frames {k0}..{k1}", flush=True)


def encode_run(run, speed=30):
    tag = run["tag"]
    outdir = TMP / f"ov_{tag}"
    out = TMP / f"overlay_{tag}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(speed), "-i", str(outdir / "o_%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    str(out)], check=True)
    print(f"[{tag}] overlay -> {out}", flush=True)
    return out


# ------------------------------------------------------------------- fitting

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


# ---------------------------------------------------------- fractal dimension

def box_count(binary, sizes):
    H, W = binary.shape
    counts = []
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        blocks = binary[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3))
        counts.append(blocks.sum())
    return np.array(counts)


def final_mask(run):
    """Deposit mask of the last frame (seed-gated)."""
    tag = run["tag"]
    ref, _ = load_ref(tag)
    frames = frames_of(tag)
    meta = json.load(open(DATA / f"meta_{tag}.json"))
    f = cv2.imread(str(frames[-1][1]))
    return cluster_gate(deposit_mask(f, ref), tuple(meta["seed"])), f



def mask_noref(bgr, hi=HYST_HI, lo=HYST_LO):
    """Per-frame deposit mask without the temporal reference: flat-field
    hysteresis, minus wire (HSV) and the blue mm-grid, despeckled, then
    disc-gated to the aggregate (components within 1.15 R99 of the centroid).
    Needed for week-4 final frames, which come after mid-run camera
    re-framing (the start-of-run reference no longer registers)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = _flatfield_bg(gray)
    score = 1.0 - gray / (bg + 1e-6)
    m = _hysteresis(score, hi, lo)
    m[wire_mask(bgr) > 0] = 0
    red = bgr[..., 2].astype(np.float32); blue = bgr[..., 0].astype(np.float32)
    blue *= 128.0 / (np.median(blue) + 1e-6)
    red *= 128.0 / (np.median(red) + 1e-6)
    m[(blue - red) > 12] = 0                    # mm-grid lines
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= MIN_SIZE
    m = keep[lab].astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 100:
        return m
    cx, cy = xs.mean(), ys.mean()
    R = np.percentile(np.hypot(xs - cx, ys - cy), 99)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = np.hypot(cent[i, 0] - cx, cent[i, 1] - cy) <= 1.15 * R
    return keep[lab].astype(np.uint8)


def fractal_run(run, thr_scan=(0.10, 0.15, 0.20)):
    """Box-counting D of the final deposit, with a threshold-scan systematic.
    Fit window: above the branch width (~4 px at 720p) and below R/8."""
    tag = run["tag"]
    ref, _ = load_ref(tag)
    frames = frames_of(tag)
    meta = json.load(open(DATA / f"meta_{tag}.json"))
    seed = tuple(meta["seed"])
    f = cv2.imread(str(frames[-1][1]))

    def _bc(m):
        ys, xs = np.nonzero(m)
        if len(xs) < 100:
            return None
        crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        R = np.percentile(np.hypot(xs - xs.mean(), ys - ys.mean()), 99)
        sizes = np.unique(np.round(
            2 ** np.arange(1, np.log2(min(crop.shape) / 2), 0.25)).astype(int))
        N = box_count(crop, sizes)
        ok = N > 0
        sizes, N = sizes[ok], N[ok]
        win = (sizes >= 4) & (sizes <= R / 8)
        if win.sum() < 4:
            win = (sizes >= 3) & (sizes <= max(R / 6, 12))
        D, dD, b = fit_loglog(sizes[win], N[win])
        return -D, dD, sizes, N, win, b, R

    # threshold-scan systematic (single-frame masks around the default)
    Ds = [r[0] for hi in thr_scan
          if (r := _bc(mask_noref(f, hi=hi, lo=hi * 0.4))) is not None]
    D_sys = float((max(Ds) - min(Ds)) / 2) if len(Ds) > 1 else 0.0

    # central value: the registered SINGLE final-frame mask (the accumulated
    # R(t) mask fattens branches with unioned jitter and biases D upward)
    m = mask_noref(f)
    D0, dD0, sizes, N, win, b, R = _bc(m)
    D_mean = float(D0)

    # mass-radius cross-check from the growth: M ~ Rg^D over unclipped growth
    rows = np.genfromtxt(DATA / f"radius_{tag}.csv", delimiter=",", names=True)
    t, M, Rg, cl = rows["t_s"], rows["npix"], rows["Rg_px"], rows["clipped"]
    Mref = np.percentile(M[M > 0], 98)
    mv = rows["moved"] if "moved" in rows.dtype.names else np.zeros_like(cl)
    w = (M > 0.08 * Mref) & (M < 0.75 * Mref) & (cl == 0) & (mv == 0) & (Rg > 0)
    w &= hampel_inliers(Rg) & hampel_inliers(M)
    Dmr, dDmr = (np.nan, np.nan)
    if w.sum() >= 6:
        Dmr, dDmr, _ = fit_loglog(Rg[w], M[w])

    # diagnostics figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    vis = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).copy()
    vis[m > 0] = (220, 30, 30)
    axs[0].imshow(vis); axs[0].axis("off")
    axs[0].set_title(f"{run['label']} — final deposit (red = mask)")
    axs[1].loglog(sizes, N, "o", ms=4, color="gray")
    axs[1].loglog(sizes[win], N[win], "o", ms=5, color="C3")
    ss = np.array([sizes[win].min(), sizes[win].max()])
    axs[1].loglog(ss, np.exp(b) * ss ** (-D0), "k-")
    axs[1].set_xlabel("box size s (px)"); axs[1].set_ylabel("N(s)")
    axs[1].set_title(f"box counting: D = {D_mean:.2f} ± "
                     f"{np.hypot(dD0, D_sys):.2f}")
    if w.sum() >= 6:
        axs[2].loglog(Rg[w], M[w], "o", ms=4, color="C2")
        rr = np.array([Rg[w].min(), Rg[w].max()])
        bmr = np.exp(np.median(np.log(M[w]) - Dmr * np.log(Rg[w])))
        axs[2].loglog(rr, bmr * rr ** Dmr, "k-")
        axs[2].set_title(f"growth mass–radius: D = {Dmr:.2f} ± {dDmr:.2f}")
    axs[2].set_xlabel("Rg (px)"); axs[2].set_ylabel("M (px)")
    fig.tight_layout()
    fig.savefig(FIGS / f"fractalD_{tag}.png", dpi=120); plt.close(fig)

    res = dict(tag=tag, conc=run["conc"], D_box=D_mean,
               dD_box=float(np.hypot(dD0, D_sys)),
               D_massradius=float(Dmr), dD_massradius=float(dDmr))
    print(f"[{tag}] D_box={D_mean:.3f}±{np.hypot(dD0, D_sys):.3f}  "
          f"D_mr={Dmr:.3f}±{dDmr:.3f}", flush=True)
    return res


# ------------------------------------------------------------------- figures

def radius_figure():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    summary = []
    for i, run in enumerate(RUNS):
        tag = run["tag"]
        rows = np.genfromtxt(DATA / f"radius_{tag}.csv", delimiter=",", names=True)
        meta = json.load(open(DATA / f"meta_{tag}.json"))
        mm = meta["mm_per_px"]
        t, R, M, cl = rows["t_s"], rows["Rfix_px"], rows["M_px"], rows["clipped"]
        # NOTE: the displayed circle radius is about the FIXED centre (offset
        # from the nucleation point), which adds a constant at early times and
        # breaks the pure power law -- beta is therefore fitted on the
        # minimum-enclosing radius Renc, as published
        Rfit = rows["Renc_px"]
        mv = rows["moved"] if "moved" in rows.dtype.names else np.zeros_like(cl)
        good = (R > 0) & (cl == 0) & (mv == 0) & hampel_inliers(R)
        clipped = (R > 0) & (cl == 1) & (mv == 0)
        axs[0].plot(t[good], R[good] * mm, ".", ms=3, color=f"C{i}",
                    label=f"{run['label']}")
        axs[0].plot(t[clipped], R[clipped] * mm, "x", ms=3, color=f"C{i}",
                    alpha=0.35)
        # power-law fit on the growth window
        t0 = nucleation_time(t, M)
        Mref = np.percentile(M[M > 0], 98)
        w = good & (M > 0.08 * Mref) & (M < 0.75 * Mref) & (t - t0 > 0)
        if w.sum() >= 6:
            beta, dbeta, b = fit_loglog(t[w] - t0, Rfit[w])
            tau = np.geomspace((t[w] - t0).min(), (t[w] - t0).max(), 50)
            axs[1].loglog(t[w] - t0, Rfit[w] * mm, ".", ms=4, color=f"C{i}",
                          label=f"{run['label']}: β = {beta:.2f} ± {dbeta:.2f}")
            axs[1].loglog(tau, np.exp(b) * tau ** beta * mm, "-", lw=1,
                          color=f"C{i}")
            summary.append(dict(tag=tag, conc=run["conc"], t0=t0,
                                beta=float(beta), dbeta=float(dbeta),
                                R_final_mm=float(np.max(R[good]) * mm)))
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("bounding-circle radius R [mm]")
    axs[0].set_title("R(t) — bounding circle about the fixed deposition centre\n"
                     "(× = deposit clipped by frame edge: lower bound)")
    axs[0].legend(fontsize=9)
    axs[1].set_xlabel("t − t₀ [s]"); axs[1].set_ylabel("R [mm]")
    axs[1].set_title("growth window, log–log:  R ∝ (t−t₀)^β  (min-enclosing R)")
    axs[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "radius_vs_time_all.png", dpi=130); plt.close(fig)
    with open(DATA / "radius_fits.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("figure -> radius_vs_time_all.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="run1,run2,run3,run4")
    ap.add_argument("--stage", default="measure",
                    choices=["seed", "measure", "merge", "figs",
                             "overlay", "encode", "fractal"])
    ap.add_argument("--k0", type=int, default=0)
    ap.add_argument("--k1", type=int, default=None)
    args = ap.parse_args()
    want = set(args.runs.split(","))
    runs = [r for r in RUNS if r["tag"] in want]
    if args.stage == "seed":
        for r in runs:
            seed_run(r)
    elif args.stage == "measure":
        for r in runs:
            measure_run(r, args.k0, args.k1)
    elif args.stage == "merge":
        for r in runs:
            merge_run(r)
    elif args.stage == "figs":
        radius_figure()
    elif args.stage == "overlay":
        for r in runs:
            overlay_run(r, args.k0, args.k1)
    elif args.stage == "encode":
        for r in runs:
            encode_run(r)
    elif args.stage == "fractal":
        out = [fractal_run(r) for r in runs]
        with open(DATA / "fractal_dimensions.json", "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
