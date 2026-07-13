#!/usr/bin/env python3
"""Salvage attempts for the 0.30 % fractal dimension — three faithful routes.

The single-snapshot box-count fails for 0.30 % (blur sigma 2.5-3 px + coarse
compact lobes leave <0.7 decade).  Three independent, validation-first routes
that do NOT relax the honesty rules:

  A. TIME-ENSEMBLE D(t).  During self-similar growth D is time-invariant, so
     ~25 independent frames each carrying a short window are collectively
     informative where one frame is not.  Early 0.30 frames are also SHARPER
     (sigma 1.7-2 vs 2.6+ at grounding).  Control: the same D(t) curve for
     the focused 0.15 % anchor must be flat at its known 1.84.

  B. SECTOR (spatial-trim) box-count.  The wire glare + specular hole corrupt
     one region; quadrants about the mask centroid that are occluder-free are
     legitimate sub-samples of a homogeneous object.  Control: the anchor's
     clean quadrants must reproduce its full-mask D.

  C. KINETIC mass-radius exponent  M ~ Rg^D  on the FAITHFUL mask across the
     growth (0.5 fps cached frames).  M and Rg are integral quantities, far
     more blur-tolerant than fine structure.  Controls: (i) kinetic D of the
     sharp 0.56 % and of 0.15 % must match their box-count values; (ii) the
     0.56 % frames synthetically blurred to 0.30 %'s sigma re-fitted -> the
     shift IS the blur bias of this estimator at 0.30 %'s blur level.

Needs the 0.5 fps frame cache from focus_scan_defocused.py (FOCUS_SCAN_SCRATCH).
"""

import csv, importlib.util, os, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "figures"; DATA = ROOT / "data"

spec = importlib.util.spec_from_file_location("fdd", HERE / "fractalD_deblur.py")
fdd = importlib.util.module_from_spec(spec)
sys.modules["fdd"] = fdd
spec.loader.exec_module(fdd)
er = fdd.er
CROP = fdd.CROP
SCRATCH = fdd.SCRATCH
FPS = 0.5

RUNS = {
    0.30: dict(vid="run 3 0.3.mov",               seed=(430, 330), ppm=47.75, t_end=138),
    0.45: dict(vid="run 2 0.45 concen.mov",       seed=(327, 367), ppm=48.16, t_end=148),
    0.56: dict(vid="run 1 0.56 Concertation.mov", seed=(463, 367), ppm=49.53, t_end=198),
    0.15: dict(vid="run4_0.15.mov",               seed=(510, 359), ppm=48.60, t_end=244),
}
BC_D = {0.15: 1.84, 0.56: 1.86, 0.45: 1.93}   # box-count references for context


def frame_at(conc, t, blur=None):
    fdir = SCRATCH / f"c{conc:.2f}"
    frames = sorted(fdir.glob("f_*.png"))
    i = min(int(round(t * FPS)), len(frames) - 1)
    img = cv2.imread(str(frames[i])).astype(np.float32)
    img = img[CROP:-CROP, CROP:-CROP]
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def prep(conc):
    run = RUNS[conc]
    ref = fdd.get_ref(run["vid"])[CROP:-CROP, CROP:-CROP]
    seed = (run["seed"][0] - CROP, run["seed"][1] - CROP)
    return run, ref, seed


def mask_of(img, ref, seed):
    m = fdd.faithful_mask(img, ref)
    m[:fdd.BORDER, :] = 0; m[-fdd.BORDER:, :] = 0
    m[:, :fdd.BORDER] = 0; m[:, -fdd.BORDER:] = 0
    return fdd.seed_cc(m, seed)


def sigma_of(img, ref, seed):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    blob, dark = fdd.blob_for_sigma(gray, ref, seed)
    return fdd.edge_sigma(gray, blob, dark) if blob is not None else np.nan


def boxcount_D(mask, sig):
    """Window-ruled box-count D of one binary mask; returns (D, decades, std)."""
    if mask.sum() < 2000:
        return np.nan, np.nan, np.nan
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    R = np.percentile(np.hypot(xs - cx, ys - cy), 99)
    w = fdd.branch_width_px(mask)
    lo = max(w, 3 * sig) if np.isfinite(sig) else w
    hi = R / 3
    if not (np.isfinite(lo) and hi > lo):
        return np.nan, np.nan, np.nan
    s, N = fdd.box_count(mask)
    ls = -fdd.local_slope(s, N)
    win = (s >= lo) & (s <= hi)
    if win.sum() < 4:
        return np.nan, np.log10(hi / lo), np.nan
    D = -np.polyfit(np.log(s[win]), np.log(N[win]), 1)[0]
    return D, np.log10(hi / lo), float(np.nanstd(ls[win]))


# ---------------------------------------------------------------- route A ---

def route_A(conc, t_lo, t_hi, step=4):
    run, ref, seed = prep(conc)
    rows = []
    for t in np.arange(t_lo, t_hi + 1, step):
        img = frame_at(conc, t)
        sig = sigma_of(img, ref, seed)
        m = mask_of(img, ref, seed)
        D, dec, sd = boxcount_D(m, sig)
        rows.append((t, sig, D, dec, sd))
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------- route B ---

def route_B(conc, t, use_stack=None):
    run, ref, seed = prep(conc)
    if use_stack:
        img, _, _ = fdd.median_stack(conc, *use_stack)
        img = img[CROP:-CROP, CROP:-CROP]
    else:
        img = frame_at(conc, t)
    sig = sigma_of(img, ref, seed)
    m = mask_of(img, ref, seed)
    occ = er.occluder_mask(img.astype(np.uint8))
    D_full, dec_full, _ = boxcount_D(m, sig)
    ys, xs = np.nonzero(m)
    cx, cy = int(xs.mean()), int(ys.mean())
    out = {"full": (D_full, dec_full, np.nan)}
    quads = {"NW": (slice(None, cy), slice(None, cx)),
             "NE": (slice(None, cy), slice(cx, None)),
             "SW": (slice(cy, None), slice(None, cx)),
             "SE": (slice(cy, None), slice(cx, None))}
    for name, (sy, sx) in quads.items():
        sub = np.zeros_like(m); sub[sy, sx] = m[sy, sx]
        # occluder contamination of this quadrant's deposit neighbourhood
        near = cv2.dilate(sub, np.ones((9, 9), np.uint8))
        occfrac = float((occ & near).sum()) / max(float(near.sum()), 1.0)
        D, dec, _ = boxcount_D(sub, sig)
        out[name] = (D, dec, occfrac)
    return out, sig


# ---------------------------------------------------------------- route C ---

def route_C(conc, blur=None, tag=None):
    """Faithful-mask kinetic fit M ~ Rg^D over the growth, 0.5 fps frames."""
    run, ref, seed = prep(conc)
    fdir = SCRATCH / f"c{conc:.2f}"
    n = len(sorted(fdir.glob("f_*.png")))
    ts, Ms, Rgs, edges = [], [], [], []
    for i in range(0, n, 2):            # every 4 s
        t = i / FPS
        img = frame_at(conc, t, blur=blur)
        m = mask_of(img, ref, seed)
        if m.sum() < 50:
            continue
        ys, xs = np.nonzero(m)
        H, W = m.shape
        edge = int(xs.min() <= fdd.BORDER + 1 or ys.min() <= fdd.BORDER + 1 or
                   xs.max() >= W - 2 - fdd.BORDER or ys.max() >= H - 2 - fdd.BORDER)
        Rg = float(np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean()))
        ts.append(t); Ms.append(float(m.sum())); Rgs.append(Rg); edges.append(edge)
    ts, Ms, Rgs, edges = map(np.array, (ts, Ms, Rgs, edges))

    def fit(lo_f, hi_f):
        ok = edges == 0
        Mref = np.percentile(Ms[ok], 98) if ok.any() else 0
        wsel = ok & (Ms >= lo_f * Mref) & (Ms <= hi_f * Mref)
        if wsel.sum() < 8:
            return np.nan, 0
        D = np.polyfit(np.log(Rgs[wsel]), np.log(Ms[wsel]), 1)[0]
        return D, int(wsel.sum())

    D0, npts = fit(0.08, 0.75)
    Ds = [fit(a, b)[0] for a, b in [(0.05, 0.85), (0.12, 0.65), (0.10, 0.70)]]
    Ds = [d for d in [D0] + Ds if np.isfinite(d)]
    sysh = (max(Ds) - min(Ds)) / 2 if len(Ds) > 1 else np.nan
    return dict(conc=conc, tag=tag or f"c{conc:.2f}", D=D0, sys=sysh, npts=npts,
                ts=ts, Ms=Ms, Rgs=Rgs, edges=edges)


def main():
    FIGS.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

    print("=== A. time-ensemble box-count D(t) ===", flush=True)
    A30 = route_A(0.30, 24, 136, step=4)
    A15 = route_A(0.15, 40, 240, step=8)
    for tag, arr in [("c0.30", A30), ("c0.15 control", A15)]:
        ok = np.isfinite(arr[:, 2])
        med = np.nanmedian(arr[ok, 2]); lo, hi = np.nanpercentile(arr[ok, 2], [16, 84])
        print(f"  {tag}: {ok.sum()} frames, D median {med:.3f} [{lo:.3f},{hi:.3f}], "
              f"decades median {np.nanmedian(arr[ok, 3]):.2f}, "
              f"sigma range [{np.nanmin(arr[:,1]):.1f},{np.nanmax(arr[:,1]):.1f}]px")
        # window-adequacy breakdown: the control shows the estimator saturates
        # at ~2.0 below ~0.5-0.6 decades regardless of the true D
        for thr in (0.4, 0.5, 0.6):
            sel = ok & (arr[:, 3] >= thr)
            if sel.sum():
                print(f"    dec>={thr}: n={sel.sum():2d}  D med "
                      f"{np.nanmedian(arr[sel, 2]):.3f}")
            else:
                print(f"    dec>={thr}: n= 0  (no adequate-window frame exists)")

    print("\n=== B. sector (quadrant) box-count ===", flush=True)
    for conc, t, stack in [(0.15, 244, None), (0.30, 138, None), (0.30, None, (300, 344))]:
        out, sig = route_B(conc, t, use_stack=stack)
        lab = f"c{conc:.2f} " + (f"stack{stack}" if stack else f"t={t}s")
        print(f"  {lab} (sigma {sig:.2f}px):")
        for k, (D, dec, occf) in out.items():
            occs = "" if k == "full" else f" occ={occf*100:4.1f}%"
            print(f"    {k:>4}: D={D:6.3f}  dec={dec:5.2f}{occs}")

    print("\n=== C. kinetic M~Rg^D on the faithful mask (every 4 s) ===", flush=True)
    kin = []
    for conc in (0.15, 0.56, 0.45, 0.30):
        r = route_C(conc)
        kin.append(r)
        bc = BC_D.get(conc)
        print(f"  c{conc:.2f}: D_kin = {r['D']:.3f} +/- {r['sys']:.3f} (sys, {r['npts']} pts)"
              + (f"   [box-count {bc}]" if bc else ""), flush=True)
    sig56 = 0.55
    s_add = np.sqrt(2.5 ** 2 - sig56 ** 2)
    rb = route_C(0.56, blur=s_add, tag="c0.56 blurred to sigma2.5")
    kin.append(rb)
    print(f"  c0.56 blurred to 0.30's sigma: D_kin = {rb['D']:.3f} +/- {rb['sys']:.3f} "
          f"-> blur bias = {rb['D'] - kin[1]['D']:+.3f}", flush=True)

    with open(DATA / "salvage_c030.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["route", "tag", "value", "err", "extra"])
        ok = np.isfinite(A30[:, 2])
        w.writerow(["A", "c0.30 D(t) median", f"{np.nanmedian(A30[ok,2]):.4f}",
                    f"{np.nanstd(A30[ok,2]):.4f}", f"n={ok.sum()}"])
        ok = np.isfinite(A15[:, 2])
        w.writerow(["A", "c0.15 control median", f"{np.nanmedian(A15[ok,2]):.4f}",
                    f"{np.nanstd(A15[ok,2]):.4f}", f"n={ok.sum()}"])
        for r in kin:
            w.writerow(["C", r["tag"], f"{r['D']:.4f}", f"{r['sys']:.4f}",
                        f"npts={r['npts']}"])

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for arr, lab, c in [(A30, "0.30 %", "C1"), (A15, "0.15 % control", "C0")]:
        ok = np.isfinite(arr[:, 2])
        ax.errorbar(arr[ok, 0], arr[ok, 2], yerr=arr[ok, 4], fmt="o-", ms=4,
                    lw=1, color=c, label=lab, alpha=0.85)
    ax.axhline(1.84, color="C0", ls=":", lw=1)
    ax.axhline(2.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("box-count D (window rule)")
    ax.set_title("A. time-ensemble D(t)"); ax.set_ylim(1.5, 2.3)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    for r in kin:
        if "blur" in r["tag"]:
            continue
        sel = r["edges"] == 0
        ax.loglog(r["Rgs"][sel], r["Ms"][sel], ".", ms=4,
                  label=f"{r['tag']}: D={r['D']:.2f}")
    ax.set_xlabel("Rg [px]"); ax.set_ylabel("M [px]")
    ax.set_title("C. kinetic M ~ Rg^D (faithful mask)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[2]
    labels, vals, errs = [], [], []
    for r in kin:
        labels.append(r["tag"].replace("c0.56 blurred to sigma2.5", "0.56 blur->2.5"))
        vals.append(r["D"]); errs.append(r["sys"])
    ax.errorbar(range(len(vals)), vals, yerr=errs, fmt="s", ms=7, capsize=4, color="C3")
    for i, c in enumerate([0.15, 0.56, 0.45, 0.30]):
        if c in BC_D:
            ax.plot(i, BC_D[c], "o", ms=9, mfc="none", mec="C0", mew=2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.axhline(2.0, color="gray", ls=":", lw=1)
    ax.set_ylabel("D"); ax.set_title("kinetic D (squares) vs box-count (circles)")
    ax.grid(alpha=0.3)
    fig.suptitle("0.30 % salvage routes: time-ensemble, sectors, kinetics", fontsize=13)
    fig.tight_layout()
    out = FIGS / "salvage_c030.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
