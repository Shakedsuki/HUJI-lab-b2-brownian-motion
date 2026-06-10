"""
focus.py  (pipeline, week2)
---------------------------
Per-frame FOCUS GATING for the MSD fit.

Why week2 needs it: buoyant polyethylene beads rise THROUGH the thin focal plane,
so a single track is sharp only in its middle and DEFOCUSES toward its ends.
Defocus degrades localization NON-stationarily (worse at the ends), which the
constant MSD intercept c cannot absorb -- it leaks into the short-lag slope and
biases D. Week1 only had a per-TRACK focus-stability gate (R_cv); here we trim
the defocused FRAMES within an otherwise-good track and keep its sharp core.

The cheap handle: FRST already writes a `sym` (symmetry/beadness) score per
detection into trajectory.csv, and sym FALLS as a bead defocuses (the ring blurs,
radial symmetry weakens). So we can gate frame-by-frame from existing columns
with NO video re-reads. `contrast` is a second proxy but flips through focus, so
sym is preferred. validate_proxy() confirms sym tracks an independent image
Laplacian-variance sharpness on sampled crops before we trust it.

Pure functions operate on a trajectory DataFrame; nothing here reads runs.json,
so it is equally usable on week1 data for validation.
"""

import numpy as np
import pandas as pd


def per_bead_focus_mask(traj, score="sym", frac=0.55, min_frac_keep=0.45,
                        smooth=5):
    """Add a boolean `in_focus` column: True where a bead's (smoothed) focus
    proxy is >= frac * its own median -- i.e. the sharp core of each track.

    Per bead, not global: every bead is sharpest at its own focal crossing, so
    the threshold is relative to that bead's median, never an absolute cut.
    `smooth` (frames) damps isolated single-frame dips so a track isn't shredded.
    A floor guarantees at least `min_frac_keep` of each track survives (else we'd
    occasionally gate away a whole faint-but-valid bead)."""
    t = traj.sort_values(["particle", "frame"]).reset_index(drop=True)
    keep = np.zeros(len(t), dtype=bool)
    for _, g in t.groupby("particle", sort=False):
        idx = g.index.to_numpy()
        s = g[score].to_numpy(float)
        if smooth > 1 and len(s) >= smooth:
            k = np.ones(smooth) / smooth
            s = np.convolve(s, k, mode="same")
        med = np.median(s)
        m = s >= frac * med if med > 0 else np.ones(len(s), bool)
        if m.mean() < min_frac_keep:                 # floor: keep top fraction
            cut = np.quantile(s, 1.0 - min_frac_keep)
            m = s >= cut
        keep[idx] = m
    t["in_focus"] = keep
    return t


def gate(traj, **kw):
    """Return only the in-focus rows (sharp core of each track)."""
    t = per_bead_focus_mask(traj, **kw)
    return t[t["in_focus"]].copy()


# ----- validation: is sym actually a focus proxy? --------------------------

def _laplacian_focus(crop):
    import cv2
    return float(cv2.Laplacian(crop.astype(np.float32), cv2.CV_32F, ksize=3).var())


def validate_proxy(traj, video_path, out_png, n_sample=300, min_len=60):
    """Confirm `sym` tracks an independent image-sharpness measure, and show the
    along-track defocus shape. Writes a 3-panel figure; returns a summary dict."""
    import cv2
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    try:
        from . import figstyle
        figstyle.set_style()
    except Exception:                                # noqa: BLE001
        pass

    counts = traj.groupby("particle")["frame"].transform("count")
    long = traj[counts >= min_len]

    # (1) sample detections across the sym range; score an image crop at each
    samp = long.sample(min(n_sample, len(long)), random_state=0)
    cap = cv2.VideoCapture(video_path)
    sym, lap = [], []
    for r in samp.itertuples(index=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, fr = cap.read()
        if not ok:
            continue
        img = fr[..., :3].mean(-1)
        h = int(max(2 * getattr(r, "r_est", 8), 12))
        x, y = int(round(r.x_raw)), int(round(r.y_raw))
        a, b = max(0, y - h), max(0, x - h)
        crop = img[a:y + h, b:x + h]
        if crop.size < 16:
            continue
        sym.append(float(r.sym)); lap.append(_laplacian_focus(crop))
    cap.release()
    sym, lap = np.array(sym), np.array(lap)
    rho = float(spearmanr(sym, lap).statistic) if len(sym) > 5 else np.nan

    # (2) along-track normalized sym profile (mean over tracks vs frac position)
    grid = np.linspace(0, 1, 21)
    prof = []
    for _, g in long.groupby("particle"):
        s = g.sort_values("frame")["sym"].to_numpy(float)
        if len(s) < min_len:
            continue
        pos = np.linspace(0, 1, len(s))
        prof.append(np.interp(grid, pos, s / (np.median(s) + 1e-9)))
    prof = np.array(prof)
    prof_mean = prof.mean(0) if len(prof) else np.full_like(grid, np.nan)

    # (3) one example track's sym vs frame with the gate overlaid
    ex_pid = long.groupby("particle")["frame"].count().idxmax()
    ex = traj[traj["particle"] == ex_pid].sort_values("frame")
    exg = per_bead_focus_mask(ex)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].scatter(sym, lap, s=12, alpha=0.4)
    ax[0].set_xlabel("FRST sym (recorded)"); ax[0].set_ylabel("image Laplacian var")
    ax[0].set_title(f"sym vs independent sharpness  (Spearman rho={rho:.2f}, "
                    f"n={len(sym)})")
    ax[1].plot(grid, prof_mean, "-o", ms=4)
    ax[1].axhline(1.0, color="0.6", ls="--", lw=1)
    ax[1].set_xlabel("fractional position along track")
    ax[1].set_ylabel("sym / track-median")
    ax[1].set_title(f"defocus at track ends  (mean over {len(prof)} tracks)")
    ax[2].plot(exg["frame"], exg["sym"], "-", color="0.6", lw=1, label="sym")
    inf = exg[exg["in_focus"]]
    ax[2].scatter(inf["frame"], inf["sym"], s=14, color="C2", label="kept (in-focus)")
    out = exg[~exg["in_focus"]]
    ax[2].scatter(out["frame"], out["sym"], s=14, color="C3", label="gated (defocus)")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("sym")
    ax[2].set_title(f"gate on longest track p{ex_pid} "
                    f"({inf.shape[0]}/{exg.shape[0]} kept)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    try:
        from . import figstyle
        p = figstyle.save(fig, out_png, dpi=120)
    except Exception:                                # noqa: BLE001
        fig.savefig(out_png, dpi=120); p = out_png
    plt.close(fig)
    return dict(spearman_rho=rho, n_proxy=len(sym), n_tracks=int(len(prof)),
                end_dip=float(prof_mean[[0, -1]].mean()) if len(prof) else np.nan,
                mid_peak=float(prof_mean[len(grid) // 2]) if len(prof) else np.nan,
                fig=p)


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser(description="Focus-gating proxy validation.")
    ap.add_argument("--traj", required=True, help="trajectory.csv path")
    ap.add_argument("--video", required=True, help="clip .avi path")
    ap.add_argument("--out", default="focus_validate.png")
    ap.add_argument("--n-sample", type=int, default=300)
    args = ap.parse_args()
    tr = pd.read_csv(args.traj)
    summ = validate_proxy(tr, args.video, args.out, n_sample=args.n_sample)
    print(f"[focus] sym<->sharpness Spearman rho = {summ['spearman_rho']:.2f} "
          f"(n={summ['n_proxy']}); along-track end/mid = "
          f"{summ['end_dip']:.2f}/{summ['mid_peak']:.2f} over {summ['n_tracks']} "
          f"tracks -> {os.path.basename(str(summ['fig']))}")
