#!/usr/bin/env python3
"""Growth kinetics of Cu electrodeposits from video (DLA experiment, Week 3).

The five still photos analysed in ``fractal_dimension.py`` give the fractal
dimension of the *final* dried deposit but carry no time axis.  These videos do:
we track the aggregate as it grows and fit the DLA growth laws

    mass    M(t) ~ (t - t0)^alpha        (alpha = 1 if deposition rate constant)
    radius  R(t) ~ (t - t0)^beta         (DLA: beta ~ 1/D ~ 0.6 when M ~ t)
    mass-radius   M ~ R^D                 (independent cross-check of D = 1.65)

Segmentation (per frame), grounded in what physically changes in the cell:
  * the deposit is the *only* thing that grows -- dish, copper-ring anode, wire,
    reflections, dust and any leftover deposit from a previous run are static, so
    a temporal "has-darkened-since-the-start" mask isolates the new growth and
    auto-removes the leftover blob present in run 2 / run 3;
  * the dendrite is traced by a flat-field local-contrast threshold (divide by a
    sigma=101 px blur, as in fractal_dimension.py) run as a hysteresis on the
    *flat-field* image -- this follows thin faint branches without bridging the
    smooth depletion/shadow halo between them into a solid blob;
  * the green/yellow cathode wire is the only moving coloured object, so it is
    segmented in HSV per frame and excluded wherever it currently is;
  * components are gated to the cluster connected to the growth seed (the cathode
    tip), dropping far dust.

R is reported in PIXELS (the exponents are scale-free; no mm calibration was
available).  Outputs: per-clip CSV in ../data/, diagnostics PNGs in ../figures/.

Run:  python3 scripts/growth_kinetics.py
Videos are large and live outside the repo; set WEEK3_VIDEO_DIR to point at them.
"""

import os
import csv
import shutil
import tempfile
import subprocess
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
VIDEO_DIR = Path(os.environ.get("WEEK3_VIDEO_DIR", r"C:\Users\Shake\Downloads\DSC_0070"))

FPS = 60000 / 1001          # 59.94 fps (from ffprobe on all clips)
SAMPLE_FPS = 3.0            # measurement cadence
REF_N = 12                  # frames median-ed for the static reference

# --- segmentation constants (fixed by the prototyping in _inspect/, not tuned
#     to any target exponent) ---
FLAT_SIGMA = 101            # flat-field blur, matches fractal_dimension.py
HYST_HI = 0.15             # strong darkness vs local bg  (= 0.85 threshold there)
HYST_LO = 0.06             # weak darkness, kept if connected to a strong pixel
CHANGE_THR = 12            # grayscale darkening vs reference to count as "grown"
S_WIRE = 100               # HSV saturation of the bright cathode wire
WIRE_DILATE = 18
MIN_SIZE = 10              # px; smaller components are speckle/dust
HUB_BRIDGE = 30            # px disk at the seed to reconnect wire-split branches
GAP_CLOSE = 4              # px close to bridge within-branch pixel gaps

CLIPS = [
    dict(tag="run1_sparse", file="run1 sparse.mov",
         label="run 1 (sparse, ~12 V)", view="wide"),
    dict(tag="run2_12V", file="run2_12V.mov",
         label="run 2 (12 V constant)", view="wide"),
    dict(tag="run3_step", file="run3_12V_to_5V6V.mov",
         label="run 3 (12 V -> 5-6 V)", view="wide", vstep=True),
    dict(tag="run4_dense", file="run 4 dense.mov",
         label="run 4 (dense, close-up)", view="close"),
    dict(tag="DSC0076", file="DSC_0076.mov",
         label="DSC_0076 (radial, close-up)", view="close"),
]


# ------------------------------------------------------------- segmentation ---

def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def wire_mask(bgr):
    """The bright green/yellow cathode wire (the only moving coloured object)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    w = ((s > S_WIRE) & (h > 18) & (h < 95) & (v > 60)).astype(np.uint8)
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
    """Smooth illumination background = sigma-blur, computed on a downscaled
    image (the background is low-frequency, so this is equivalent to a full-res
    sigma-blur but ~50x faster -- the per-frame cost that matters over a video)."""
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sb = cv2.GaussianBlur(small, (0, 0), sigma * scale)
    return cv2.resize(sb, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def deposit_mask(bgr, ref_gray, hi=HYST_HI, lo=HYST_LO):
    """New dark dendrite pixels = flat-field-dark AND darkened-since-start AND
    not-wire, despeckled."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = _flatfield_bg(gray)
    score = 1.0 - gray / (bg + 1e-6)                 # local-contrast darkness
    m = _hysteresis(score, hi, lo)
    offset = np.median(gray) - np.median(ref_gray)   # kill auto-exposure drift
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
    """Keep only the dendrite physically connected to the cathode tip (seed).

    A small disk at the seed reconnects branches that the masked wire hub splits
    apart (radial stars), and a light morphological close bridges within-branch
    pixel gaps.  Far objects -- a leftover deposit from a previous run, surface
    reflections -- stay disconnected from the seed and are dropped, which is what
    keeps R(t) from jumping when the cluster nearly touches such an object."""
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


# ---------------------------------------------------------- video iteration ---

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def extract_frames(path, tmp):
    """Decode the clip with ffmpeg (far faster than per-frame OpenCV): the first
    REF_N frames for the reference, plus frames resampled to SAMPLE_FPS.  Returns
    (ref_gray, [(t_seconds, frame_path), ...])."""
    refdir, samp = tmp / "ref", tmp / "f"
    refdir.mkdir(); samp.mkdir()
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-frames:v", str(REF_N), str(refdir / "r_%03d.png")], check=True)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-vf", f"fps={SAMPLE_FPS}", "-q:v", "2", str(samp / "f_%05d.jpg")],
                   check=True)
    refs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for p in sorted(refdir.glob("r_*.png"))]
    ref = np.median(np.stack(refs), axis=0)
    frames = [(k / SAMPLE_FPS, p) for k, p in enumerate(sorted(samp.glob("f_*.jpg")))]
    return ref, frames


def find_seed(frames, ref):
    """Growth seed = the cathode tip: the persistent early-deposit pixel CLOSEST
    TO THE WIRE (where the dendrite roots).  Anchoring to the wire, rather than to
    the densest accumulation, keeps the seed at the true growth origin for
    one-sided fans as well as radial stars."""
    acc = np.zeros(ref.shape, np.float32)
    wu = np.zeros(ref.shape, np.uint8)
    early = frames[:max(int(0.30 * len(frames)), 3)]
    for _, p in early:
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


def measure(frames, ref, seed, save_overlays_at=()):
    """Measure R(t), M(t) on every extracted frame."""
    rows, overlays = [], {}
    sx, sy = seed
    nfr = len(frames)
    for k, (t, p) in enumerate(frames):
        f = cv2.imread(str(p))
        tree = cluster_gate(deposit_mask(f, ref), seed)
        ys, xs = np.nonzero(tree)
        if len(xs) < 5:
            rows.append((t, 0, 0.0, 0.0, 0.0, 0))
            continue
        d = np.hypot(xs - sx, ys - sy)
        cx, cy = xs.mean(), ys.mean()
        rg = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean()))
        ncomp = int(cv2.connectedComponents(tree)[0] - 1)
        rows.append((t, int(tree.sum()), float(np.percentile(d, 95)),
                     float(np.percentile(d, 99.5)), rg, ncomp))
        for frac in save_overlays_at:
            key = round(frac, 3)
            if key not in overlays and k >= frac * (nfr - 1):
                vis = f.copy()
                vis[tree > 0] = (0, 220, 0)
                cv2.circle(vis, seed, 6, (0, 255, 255), -1)
                cv2.circle(vis, seed, int(np.percentile(d, 95)), (0, 255, 255), 2)
                overlays[key] = (t, vis)
    return np.array(rows, dtype=float), overlays


# ----------------------------------------------------------------- fitting ---

def fit_loglog(x, y):
    """slope, stderr, intercept of log y vs log x."""
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    dof = max(len(lx) - 2, 1)
    rv = (res[0] / dof) if len(res) else 0.0
    cov = rv * np.linalg.inv(A.T @ A)
    return coef[0], float(np.sqrt(cov[0, 0])), coef[1]


def hampel_inliers(y, k=5, nsig=3.5):
    """Flag isolated non-physical spikes: the deposit grows smoothly, so a point
    far from its local median (transient reflection/occlusion) is rejected."""
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
    """Observed nucleation t0 = first frame whose mass reaches ``frac`` of the
    plateau.  Fixing t0 (rather than fitting it freely, which trades off against
    the exponent and is sampling-dependent) is what makes alpha, beta and D
    mutually consistent."""
    Mref = np.percentile(M[M > 0], 98) if np.any(M > 0) else 0.0
    hit = np.where(M >= frac * Mref)[0]
    return float(t[hit[0]]) if len(hit) else 0.0


def growth_window(t, M, R, lo_frac=0.08, hi_frac=0.75):
    """Objective fit window = the self-similar GROWTH regime: deposit established
    (M above lo_frac of the plateau) and still rising before saturation (M below
    hi_frac of the plateau), with isolated spikes removed.  The upper cut matters
    here: at constant voltage the current decays, so M saturates while the few
    surviving tips keep creeping outward -- past that point M and R decouple and
    the cluster is no longer self-similar.  The plateau uses the 98th percentile
    of M so one spurious frame cannot set the scale.  No expected exponent enters."""
    Mref = np.percentile(M[M > 0], 98) if np.any(M > 0) else 0.0
    w = (M >= lo_frac * Mref) & (M <= hi_frac * Mref) & (R > 0) & (M > 0)
    w &= hampel_inliers(R) & hampel_inliers(M)
    return w


def fit_kinetics(t, M, R95, Rg, w, t0):
    """Power-law slopes over the growth window with t measured from nucleation.
    Radius of gyration Rg is the primary radius (R95/the reach saturates at the
    frame edge); M ~ Rg^D is the standard mass-radius fractal dimension."""
    tau = t - t0
    m = w & (tau > 0)
    a, da, _ = fit_loglog(tau[m], M[m])
    b, db, _ = fit_loglog(tau[m], Rg[m])
    b95, db95, _ = fit_loglog(tau[m], R95[m])
    D, dD, _ = fit_loglog(Rg[m], M[m])
    D95, dD95, _ = fit_loglog(R95[m], M[m])
    return dict(alpha=a, dalpha=da, beta=b, dbeta=db, beta95=b95, dbeta95=db95,
                D=D, dD=dD, D95=D95, dD95=dD95, n_fit=int(m.sum()))


def window_systematic(t, M, R95, Rg, t0):
    """Spread of alpha, beta, D across reasonable growth-window choices -- a
    systematic that dominates the (tiny) statistical slope errors here."""
    wins = [(0.08, 0.75), (0.05, 0.85), (0.12, 0.65), (0.10, 0.70)]
    a_, b_, D_ = [], [], []
    for lo, hi in wins:
        w = growth_window(t, M, Rg, lo, hi)
        tau = t - t0
        m = w & (tau > 0)
        if m.sum() < 6:
            continue
        a_.append(fit_loglog(tau[m], M[m])[0])
        b_.append(fit_loglog(tau[m], Rg[m])[0])
        D_.append(fit_loglog(Rg[m], M[m])[0])
    half = lambda x: (max(x) - min(x)) / 2 if len(x) > 1 else 0.0
    return dict(alpha_sys=half(a_), beta_sys=half(b_), D_sys=half(D_))


# --------------------------------------------------------------- pipeline ----

def process(clip):
    path = VIDEO_DIR / clip["file"]
    tmp = Path(tempfile.mkdtemp(prefix=f"gk_{clip['tag']}_"))
    try:
        ref, frames = extract_frames(path, tmp)
        seed = find_seed(frames, ref)
        rows, overlays = measure(frames, ref, seed,
                                 save_overlays_at=(0.12, 0.5, 0.97))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    t, M, R95, R995, Rg, ncomp = rows.T
    DATA.mkdir(exist_ok=True)
    with open(DATA / f"kinetics_{clip['tag']}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "M_px", "R95_px", "R99.5_px", "Rg_px", "n_comp"])
        wr.writerows(rows)

    t0 = nucleation_time(t, M)
    w = growth_window(t, M, Rg)
    res = dict(clip=clip, t=t, M=M, R95=R95, R995=R995, Rg=Rg, seed=seed,
               overlays=overlays, win=w, t0=t0)
    if w.sum() >= 6:
        res.update(fit_kinetics(t, M, R95, Rg, w, t0))
        res.update(window_systematic(t, M, R95, Rg, t0))
    return res


# ----------------------------------------------------------------- figures ---

def _tot(stat, sys):
    return float(np.hypot(stat, sys))


def clip_figure(res):
    clip = res["clip"]
    t, M, R95, Rg, w = res["t"], res["M"], res["R95"], res["Rg"], res["win"]
    ov = res["overlays"]
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

    for i, frac in enumerate(sorted(ov)):
        ax = fig.add_subplot(gs[0, i])
        tt, vis = ov[frac]
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t = {tt:.0f} s   (green = deposit, R95 circle)", fontsize=9)
        ax.axis("off")

    # radius (Rg, R95) and mass vs time, fit window highlighted
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, Rg, ".", color="C2", ms=3, label="Rg (gyration)")
    ax.plot(t, R95, ".", color="C0", ms=3, alpha=0.5, label="R95 (reach)")
    ax.plot(t[w], Rg[w], "o", color="C3", ms=3, label="fit window")
    ax.set_xlabel("t [s]"); ax.set_ylabel("radius [px]")
    ax.set_title("radius & mass vs time"); ax.legend(fontsize=8, loc="upper left")
    axM = ax.twinx()
    axM.plot(t, M, ".", color="C1", ms=2, alpha=0.4)
    axM.set_ylabel("M [px]", color="C1")

    # Rg ~ (t-t0)^beta
    ax = fig.add_subplot(gs[1, 1])
    if "beta" in res:
        tau = t - res["t0"]
        mm = w & (tau > 0)
        ax.loglog(tau[mm], Rg[mm], "o", ms=4, color="C2")
        xs = np.array([tau[mm].min(), tau[mm].max()])
        A = np.exp(np.median(np.log(Rg[mm]) - res["beta"] * np.log(tau[mm])))
        ax.loglog(xs, A * xs ** res["beta"], "k-",
                  label=f"beta = {res['beta']:.2f} +/- {_tot(res['dbeta'], res['beta_sys']):.2f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("t - t0 [s]"); ax.set_ylabel("Rg [px]")
    ax.set_title("radius of gyration  Rg ~ (t-t0)^beta")

    # M ~ Rg^D
    ax = fig.add_subplot(gs[1, 2])
    if "D" in res:
        mm = w
        ax.loglog(Rg[mm], M[mm], "o", ms=4, color="C2")
        rr = np.array([Rg[mm].min(), Rg[mm].max()])
        b = np.exp(np.median(np.log(M[mm]) - res["D"] * np.log(Rg[mm])))
        ax.loglog(rr, b * rr ** res["D"], "k-",
                  label=f"D = {res['D']:.2f} +/- {_tot(res['dD'], res['D_sys']):.2f}")
        ax.legend(fontsize=9)
        ax.text(0.05, 0.95, f"alpha(M~t) = {res['alpha']:.2f}\n"
                f"alpha/beta = {res['alpha']/res['beta']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_xlabel("Rg [px]"); ax.set_ylabel("M [px]")
    ax.set_title("mass-radius  M ~ Rg^D")

    fig.suptitle(f"{clip['label']}  -  growth kinetics", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGS / f"kinetics_{clip['tag']}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def summary_figure(results):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, r in enumerate(results):
        t, Rg, w = r["t"], r["Rg"], r["win"]
        ax[0].plot(t, Rg, ".", ms=2.5, color=f"C{i}", label=r["clip"]["tag"])
        if "D" in r:
            ax[1].loglog(Rg[w], r["M"][w], "o", ms=2.5, color=f"C{i}",
                         label=f"{r['clip']['tag']}  D={r['D']:.2f}")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("Rg [px]")
    ax[0].set_title("aggregate radius of gyration vs time"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("Rg [px]"); ax[1].set_ylabel("M [px]")
    ax[1].set_title("mass-radius  M ~ Rg^D"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / "kinetics_summary.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    global SAMPLE_FPS
    import argparse
    ap = argparse.ArgumentParser(description="DLA growth kinetics from video")
    ap.add_argument("--clips", default="", help="comma-separated tags to run (default: all)")
    ap.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    args = ap.parse_args()
    SAMPLE_FPS = args.sample_fps
    want = set(t.strip() for t in args.clips.split(",") if t.strip())
    clips = [c for c in CLIPS if not want or c["tag"] in want]

    FIGS.mkdir(exist_ok=True)
    results = []
    for clip in clips:
        print(f"[{clip['tag']}] processing {clip['file']} ...", flush=True)
        r = process(clip)
        results.append(r)
        if "D" in r:
            print(f"  seed={r['seed']}  t0={r['t0']:.1f}s  n_fit={r['n_fit']}  "
                  f"alpha(M~t)={r['alpha']:.2f}+/-{_tot(r['dalpha'], r['alpha_sys']):.2f}  "
                  f"beta(Rg~t)={r['beta']:.2f}+/-{_tot(r['dbeta'], r['beta_sys']):.2f}  "
                  f"D(M~Rg)={r['D']:.2f}+/-{_tot(r['dD'], r['D_sys']):.2f}  "
                  f"alpha/beta={r['alpha']/r['beta']:.2f}  "
                  f"[D(M~R95)={r['D95']:.2f} beta95={r['beta95']:.2f}]", flush=True)
        else:
            print("  (insufficient growth window)", flush=True)
        clip_figure(r)
    summary_figure(results)
    print(f"\nfigures -> {FIGS}", flush=True)
    return results


if __name__ == "__main__":
    main()
