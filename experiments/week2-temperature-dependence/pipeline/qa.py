"""
qa.py  (pipeline, week2)
------------------------
Pre-flight CLIP HEALTH pass -- runs BEFORE any expensive tracking, on every clip,
to triage the 16 week2 runs and to PROPOSE the settled/clean analysis window that
goes back into runs.json (window=[f0,f1]). Week2 has harder data (out-of-focus
stretches, equilibration transients, thermal convection), so we look before we
track.

Signals, all from a sparse set of evenly-spaced sample frames (cheap; no full
link):
  focus(t)   : variance of the Laplacian of the (flat-fielded) frame -- a global
               sharpness score. Defocus stretches (focus-knob drift, condensation
               on a cold cell) show as sustained dips.
  bright(t)  : mean raw intensity -- illumination stability / condensation.
  count(t)   : number of FRST detections -- beads entering/leaving + a focus proxy
               (defocus collapses the count).
  flow(t)    : MEDIAN bead displacement f->f+1 (px/frame). The collective drift
               RATE vs time -- a settling stage or a leak shows here.
  convect(t) : the part of the displacement field EXPLAINED BY POSITION. We fit
               d_i ~= A (p_i - p_bar) + b per frame; b is uniform drift (already in
               flow), A is the position-dependent part -- shear / rotation /
               divergence, i.e. the signature of a CONVECTION ROLL that a single
               global drift-subtraction cannot remove. Reported as the RMS
               structured speed [px/frame] and the affine R^2.

Outputs per run:  pipeline/qa.json (scalar summary + proposed window) and
                  pipeline/qa_health.png (the one-page sheet, incl. a flow quiver
                  so convection rolls are visible by eye).

Usage:  python -m pipeline.qa run3 [--n-sample 60]
        python -m pipeline.qa --all
"""

import json
import os
import numpy as np
import pandas as pd

from . import detect, frames as fr, paths


# ----- per-frame primitives ------------------------------------------------

def _laplacian_var(img):
    """Global sharpness: variance of the Laplacian. Higher = sharper."""
    import cv2
    return float(cv2.Laplacian(img, cv2.CV_32F, ksize=3).var())


def _link_pair(d0, d1, search=8.0):
    """Nearest-neighbour match detections of frame f (d0) to f+1 (d1) within
    `search` px. Returns (p0 [K,2], disp [K,2]) for matched beads."""
    from scipy.spatial import cKDTree
    if not len(d0["x"]) or not len(d1["x"]):
        return np.empty((0, 2)), np.empty((0, 2))
    p0 = np.column_stack([d0["x"], d0["y"]])
    p1 = np.column_stack([d1["x"], d1["y"]])
    dist, idx = cKDTree(p1).query(p0, k=1)
    ok = dist <= search
    return p0[ok], (p1[idx[ok]] - p0[ok])


def _affine_flow(p0, disp):
    """Decompose a displacement field into uniform drift + position-dependent
    (convective) part:  disp ~= A (p0 - p_bar) + b.

    Returns dict: drift (b, px/frame), struct_rms (RMS of A-part, px/frame),
    r2 (variance fraction explained by position beyond translation), div, curl,
    shear (per-frame field descriptors, px/frame across the FOV)."""
    K = len(p0)
    out = dict(drift=np.array([np.nan, np.nan]), struct_rms=np.nan, r2=np.nan,
               div=np.nan, curl=np.nan, shear=np.nan, n=K)
    if K < 6:
        if K:
            out["drift"] = np.median(disp, axis=0)
        return out
    pbar = p0.mean(0)
    X = np.column_stack([p0 - pbar, np.ones(K)])        # [K,3]: dx, dy, 1
    coef, *_ = np.linalg.lstsq(X, disp, rcond=None)     # [3,2]
    A = coef[:2, :].T                                   # [2,2], rows=output dim
    b = coef[2, :]
    struct = (p0 - pbar) @ A.T                          # position-dependent part
    resid = disp - (struct + b)
    sst = float(((disp - disp.mean(0)) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else np.nan
    # scale field descriptors to a FOV-spanning displacement (per ~600 px)
    span = 600.0
    out.update(drift=b, struct_rms=float(np.sqrt((struct ** 2).sum(1).mean())),
               r2=r2, div=float((A[0, 0] + A[1, 1]) * span),
               curl=float((A[1, 0] - A[0, 1]) * span),
               shear=float(np.hypot(A[0, 0] - A[1, 1], A[0, 1] + A[1, 0]) * span),
               n=K)
    return out


# ----- the pass ------------------------------------------------------------

def scan(stem, videos_dir=None, n_sample=40, search=8.0, downscale=2,
         detect_kw=None):
    """Sparse health scan. Returns a per-sample DataFrame + a quiver snapshot.

    Detection runs on 1/downscale-resolution frames -- FRST is ~O(pixels), so
    half-res is ~4x faster and is plenty for COUNTS + FLOW + the convection affine
    fit (we don't need curation-grade localization here). Positions/displacements
    are scaled back to full-res px before reporting."""
    import cv2
    detect_kw = detect_kw or dict(sym_min=0.18, grad_pct=80.0,
                                  radii=np.arange(2, 11, 2), min_sep=3)
    vid = paths.video(paths.video_for_run(stem), videos_dir)
    out = paths.out_dir(stem)
    n_tot = fr.count_frames(vid)
    print(f"[qa] {stem}: {vid} ({n_tot} frames); flat-field + scan {n_sample} "
          f"samples @ 1/{downscale} res", flush=True)
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"), n_sample=60)

    def ds(img):
        return cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale),
                          interpolation=cv2.INTER_AREA)

    # sample frames evenly, but never the very last (need f+1 for flow)
    idxs = np.unique(np.linspace(0, max(n_tot - 2, 0), n_sample).astype(int))
    rows, quiver = [], None
    mid_target = idxs[len(idxs) // 2]
    for k, f0 in enumerate(idxs):
        a = fr.frame_at(vid, int(f0))
        b = fr.frame_at(vid, int(f0) + 1)
        if a is None or b is None:
            continue
        ia, ib = ds(a - flat), ds(b - flat)
        da = detect.detect_frame(ia, **detect_kw)
        db = detect.detect_frame(ib, **detect_kw)
        p0, disp = _link_pair(da, db, search=search / downscale)
        p0, disp = p0 * downscale, disp * downscale       # back to full-res px
        aff = _affine_flow(p0, disp)
        rows.append(dict(
            frame=int(f0),
            focus=_laplacian_var(ia),
            bright=float(a.mean()),
            count=int(len(da["x"])),
            n_link=int(len(p0)),
            flow=float(np.hypot(*aff["drift"])) if np.all(np.isfinite(aff["drift"])) else np.nan,
            struct_rms=aff["struct_rms"], affine_r2=aff["r2"],
            div=aff["div"], curl=aff["curl"], shear=aff["shear"]))
        if f0 == mid_target and len(p0) >= 6:
            quiver = (a, p0, disp, aff)
        if (k + 1) % 10 == 0:
            print(f"    [qa] {k + 1}/{len(idxs)} samples", flush=True)
    df = pd.DataFrame(rows)
    return df, quiver, n_tot


def propose_window(df, n_tot, focus_floor=0.55, flow_ceil_px=3.0):
    """Longest contiguous run of samples that are IN FOCUS (focus >= floor *
    median) and not flow-dominated. Returns ([f0,f1] or None, note)."""
    if df.empty:
        return None, "no samples"
    fmed = df["focus"].median()
    good = (df["focus"] >= focus_floor * fmed) & (
        df["flow"].fillna(0) <= flow_ceil_px)
    if not good.any():
        return None, "no in-focus low-flow samples"
    # longest True run over the sampled sequence
    g = good.to_numpy().astype(int)
    best_len = best_i = cur_len = cur_i = 0
    for i, v in enumerate(g):
        if v:
            if cur_len == 0:
                cur_i = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_i = cur_len, cur_i
        else:
            cur_len = 0
    f_lo = int(df["frame"].iloc[best_i])
    f_hi = int(df["frame"].iloc[best_i + best_len - 1])
    frac = best_len / len(df)
    if f_lo <= int(df["frame"].iloc[0]) and f_hi >= int(df["frame"].iloc[-1]):
        return None, f"full clip clean ({frac:.0%} good)"
    return [f_lo, f_hi], f"settled window {frac:.0%} of clip"


def summarize(stem, df, n_tot, window, note):
    fmed = float(df["focus"].median())
    flags = []
    if (df["focus"] < 0.5 * fmed).mean() > 0.15:
        flags.append("defocus-stretches")
    if df["flow"].max() > 5.0:
        flags.append("high-drift")
    if df["struct_rms"].median() > 1.0 or (df["affine_r2"] > 0.25).mean() > 0.3:
        flags.append("convection")
    if (df["bright"].max() - df["bright"].min()) / (df["bright"].mean() + 1e-9) > 0.15:
        flags.append("illumination-drift")
    if df["count"].median() < 8:
        flags.append("few-beads")
    return dict(
        run=stem, n_frames=int(n_tot), n_samples=int(len(df)),
        focus_med=fmed, focus_cv=float(df["focus"].std() / (fmed + 1e-9)),
        bright_cv=float(df["bright"].std() / (df["bright"].mean() + 1e-9)),
        count_med=float(df["count"].median()),
        flow_med_px=float(df["flow"].median()), flow_max_px=float(df["flow"].max()),
        struct_rms_med_px=float(df["struct_rms"].median()),
        affine_r2_med=float(df["affine_r2"].median()),
        proposed_window=window, window_note=note,
        flags=flags, verdict=("clean" if not flags else ";".join(flags)))


def render(stem, df, quiver, summ, out_path):
    import matplotlib.pyplot as plt
    from . import figstyle
    figstyle.set_style()
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fmed = summ["focus_med"]
    t = df["frame"]
    win = summ["proposed_window"]

    def shade(a):
        if win:
            a.axvspan(win[0], win[1], color="C2", alpha=0.10, zorder=0)

    ax[0, 0].plot(t, df["focus"], "-o", ms=3, color="C0"); shade(ax[0, 0])
    ax[0, 0].axhline(fmed, color="0.6", ls="--", lw=1)
    ax[0, 0].axhline(0.5 * fmed, color="C3", ls=":", lw=1, label="0.5x median")
    ax[0, 0].set_title("focus (Laplacian var) vs frame"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_xlabel("frame"); ax[0, 0].set_ylabel("sharpness")

    ax[0, 1].plot(t, df["bright"], "-o", ms=3, color="C1"); shade(ax[0, 1])
    ax[0, 1].set_title(f"brightness (CV={summ['bright_cv']:.2%})")
    ax[0, 1].set_xlabel("frame"); ax[0, 1].set_ylabel("mean intensity")

    ax[0, 2].plot(t, df["count"], "-o", ms=3, color="C4", label="detections")
    ax[0, 2].plot(t, df["n_link"], "-o", ms=2, color="0.6", label="linked f->f+1")
    shade(ax[0, 2])
    ax[0, 2].set_title(f"bead count (med={summ['count_med']:.0f})")
    ax[0, 2].set_xlabel("frame"); ax[0, 2].legend(fontsize=8)

    ax[1, 0].plot(t, df["flow"], "-o", ms=3, color="C5")
    ax[1, 0].axhline(3.0, color="C3", ls=":", lw=1, label="3 px/frame"); shade(ax[1, 0])
    ax[1, 0].set_title(f"collective drift rate (med={summ['flow_med_px']:.2f} px/fr)")
    ax[1, 0].set_xlabel("frame"); ax[1, 0].set_ylabel("|median step| px/frame")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(t, df["struct_rms"], "-o", ms=3, color="C6", label="struct RMS px/fr")
    axb = ax[1, 1].twinx()
    axb.plot(t, df["affine_r2"], "-s", ms=2, color="0.5", alpha=0.7)
    axb.set_ylabel("affine R^2", color="0.5"); axb.set_ylim(0, 1)
    shade(ax[1, 1])
    ax[1, 1].set_title(f"CONVECTION: structured flow "
                       f"(med={summ['struct_rms_med_px']:.2f} px/fr, "
                       f"R2={summ['affine_r2_med']:.2f})")
    ax[1, 1].set_xlabel("frame"); ax[1, 1].set_ylabel("struct RMS px/fr", color="C6")
    ax[1, 1].legend(fontsize=8, loc="upper left")

    # quiver: displacement field at a mid clip frame -> convection rolls visible
    if quiver is not None:
        img, p0, disp, aff = quiver
        ax[1, 2].imshow(img, cmap="gray")
        mag = np.hypot(disp[:, 0], disp[:, 1])
        ax[1, 2].quiver(p0[:, 0], p0[:, 1], disp[:, 0], -disp[:, 1], mag,
                        cmap="plasma", scale=60, width=0.004)
        ax[1, 2].set_title(f"flow field @mid (div={aff['div']:.1f} curl={aff['curl']:.1f} "
                           f"shear={aff['shear']:.1f} px/FOV)")
    else:
        ax[1, 2].text(0.5, 0.5, "no quiver (too few linked beads)",
                      ha="center", va="center")
    ax[1, 2].axis("off")

    wtxt = f"window {win}" if win else "full clip"
    fig.suptitle(f"{stem}: QA health  |  {summ['n_frames']} frames  |  "
                 f"VERDICT: {summ['verdict']}  |  proposed {wtxt}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = figstyle.save(fig, out_path, dpi=120)
    plt.close(fig)
    return p


def run(stem, videos_dir=None, n_sample=60):
    out = paths.out_dir(stem)
    df, quiver, n_tot = scan(stem, videos_dir=videos_dir, n_sample=n_sample)
    if df.empty:
        print(f"[qa] {stem}: no frames scanned")
        return None
    window, note = propose_window(df, n_tot)
    summ = summarize(stem, df, n_tot, window, note)
    df.to_csv(os.path.join(out, "qa_scan.csv"), index=False)
    with open(os.path.join(out, "qa.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    p = render(stem, df, quiver, summ, os.path.join(out, "qa_health.png"))
    print(f"[qa] {stem}: VERDICT={summ['verdict']}; focus_cv={summ['focus_cv']:.2f} "
          f"flow_med={summ['flow_med_px']:.2f} struct_rms={summ['struct_rms_med_px']:.2f} "
          f"px/fr; window={window} ({note})")
    print(f"[qa] wrote qa.json + qa_scan.csv + {os.path.basename(p)} -> {out}")
    return summ


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Pre-flight clip health QA.")
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-sample", type=int, default=60)
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()

    if args.all:
        runs = sorted(paths.load_runs().get("runs", {}),
                      key=lambda s: int(s[3:]))
        summ = [run(r, videos_dir=args.videos_dir, n_sample=args.n_sample)
                for r in runs]
        summ = [s for s in summ if s]
        tbl = pd.DataFrame(summ)[
            ["run", "n_frames", "focus_cv", "count_med", "flow_med_px",
             "struct_rms_med_px", "affine_r2_med", "verdict"]]
        print("\n=== QA SUMMARY (all runs) ===")
        print(tbl.to_string(index=False))
        tbl.to_csv(os.path.join(paths.FIGURES_DIR, "qa_summary.csv"), index=False)
        print(f"\n[qa] wrote qa_summary.csv -> {paths.FIGURES_DIR}")
    else:
        run(args.run or "run3", videos_dir=args.videos_dir, n_sample=args.n_sample)


if __name__ == "__main__":
    main()
