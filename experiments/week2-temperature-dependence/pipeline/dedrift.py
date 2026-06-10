"""
dedrift.py  (pipeline, week2)
-----------------------------
Local (AFFINE) drift subtraction to remove thermal CONVECTION -- the spatially
varying flow that the global median-translation de-drift in track.py cannot.

track.robust_drift removes the UNIFORM part of the collective flow (the median
bead step). A convection roll driven by a thermal gradient is position-dependent:
after removing the uniform part a residual v(x,y).t survives, masquerades as
enhanced diffusion, and biases D high (worst for big beads at long lags). We fit,
per frame, the increment field

        d_i  ~=  A (p_i - p_bar) + b           (b = uniform drift, already handled;
                                                A = shear / rotation / divergence)

and subtract BOTH, re-integrating each bead from its raw anchor frame.

THE OVER-SUBTRACTION TRAP (and the guard): a flexible field fit to bead steps can
soak up genuine Brownian motion and bias D LOW -- the opposite error. So the
position-dependent part A is applied to a frame ONLY when it is statistically
supported: enough beads present AND the structured RMS exceeds a shuffled-position
null (permuting which position goes with which step kills any real spatial flow
but preserves the step distribution). Otherwise we fall back to b-only. On a
convection-free clip this therefore reduces, by construction, to the existing
global de-drift -- it can only help, not silently bias.

Pure-DataFrame; no runs.json dependency (usable on week1 data for validation).
"""

import numpy as np
import pandas as pd


def _steps(traj):
    """Consecutive-frame steps. Returns a DataFrame: particle, frame (=f0),
    x,y (raw position at f0), dx,dy (increment to f0+1)."""
    t = traj.sort_values(["particle", "frame"])
    gx = t.groupby("particle")
    dx = gx["x_raw"].diff().shift(-1)
    dy = gx["y_raw"].diff().shift(-1)
    df = gx["frame"].diff().shift(-1)
    s = pd.DataFrame({"particle": t["particle"], "frame": t["frame"],
                      "x": t["x_raw"], "y": t["y_raw"], "dx": dx, "dy": dy,
                      "dfr": df})
    return s[s["dfr"] == 1].drop(columns="dfr").reset_index(drop=True)


def fit_affine(p0, d):
    """Least-squares  d ~= A (p0 - pbar) + b. Returns A[2,2], b[2], struct (the
    A-part per bead), resid, r2, struct_rms."""
    pbar = p0.mean(0)
    X = np.column_stack([p0 - pbar, np.ones(len(p0))])
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    A, b = coef[:2].T, coef[2]
    struct = (p0 - pbar) @ A.T
    resid = d - (struct + b)
    sst = float(((d - d.mean(0)) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else np.nan
    return A, b, struct, resid, r2, float(np.sqrt((struct ** 2).sum(1).mean()))


def _null_struct_rms(p0, d, n_perm=24, seed=0):
    """95th-percentile structured RMS under permuted position<->step pairing
    (destroys real spatial flow, keeps the step distribution)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation(len(p0))
        _, _, _, _, _, srms = fit_affine(p0[perm], d)
        vals.append(srms)
    return float(np.percentile(vals, 95))


def frame_models(traj, min_n=12, n_perm=24, drop_jumps=True, jump_k=6.0):
    """Per-frame flow model. For each frame f: uniform drift b (median step) and,
    where SIGNIFICANT (>= min_n beads AND structured RMS above the shuffled-null
    95th pct), the convective affine part A about that frame's centroid pbar.
    Returns {frame: dict(A, b, pbar, applied, srms, null, n)}."""
    s = _steps(traj)
    models = {}
    for f, g in s.groupby("frame"):
        p0 = g[["x", "y"]].to_numpy(float)
        d = g[["dx", "dy"]].to_numpy(float)
        if drop_jumps and len(d) >= 5:               # drop gross mislinks from the fit
            mag = np.hypot(d[:, 0], d[:, 1])
            med = np.median(mag)
            mad = np.median(np.abs(mag - med)) * 1.4826 + 1e-9
            keep = mag <= med + jump_k * mad
            p0, d = p0[keep], d[keep]
        m = dict(A=None, b=np.zeros(2), pbar=np.zeros(2), applied=False,
                 srms=np.nan, null=np.nan, n=int(len(p0)))
        if len(p0) >= 3:
            m["b"] = np.median(d, axis=0)
            m["pbar"] = p0.mean(0)
            if len(p0) >= min_n:
                A_, b_, _, _, _, srms = fit_affine(p0, d)
                nul = _null_struct_rms(p0, d, n_perm=n_perm)
                m.update(srms=srms, null=nul)
                if srms > nul:                       # convection is real this frame
                    m.update(A=A_, b=b_, applied=True)
        models[int(f)] = m
    return models


def local_dedrift(traj, min_n=12, n_perm=24, log=False, models=None):
    """De-drift removing uniform + (where significant) convective flow, by
    integrating each bead's corrected increments from its raw anchor frame. Gaps
    integrate the uniform part only.

    KEY PROPERTY: on a convection-free clip no frame passes the null test, so
    every correction collapses to b (the median step) and this reduces EXACTLY to
    the global de-drift -- it can only remove real convection, never add bias.

    Returns a copy of traj with x,y replaced (x_raw,y_raw preserved); if log, also
    a per-frame DataFrame."""
    if models is None:
        models = frame_models(traj, min_n=min_n, n_perm=n_perm)
    t = traj.sort_values(["particle", "frame"]).copy()
    if "x_raw" not in t:
        t["x_raw"], t["y_raw"] = t["x"], t["y"]
    xs, ys = np.empty(len(t)), np.empty(len(t))
    i = 0
    for _, g in t.groupby("particle", sort=False):
        f = g["frame"].to_numpy(int)
        rx = g["x_raw"].to_numpy(float)
        ry = g["y_raw"].to_numpy(float)
        cx, cy = rx[0], ry[0]                         # anchor at raw start
        xs[i], ys[i] = cx, cy
        for k in range(1, len(f)):
            if f[k] - f[k - 1] == 1:
                m = models.get(int(f[k - 1]))
                if m is None:
                    off = np.zeros(2)
                elif m["A"] is not None:
                    off = m["A"] @ (np.array([rx[k - 1], ry[k - 1]]) - m["pbar"]) + m["b"]
                else:
                    off = m["b"]
                offx, offy = float(off[0]), float(off[1])
            else:                                     # gap: integrate uniform part only
                offx = offy = 0.0
                for j in range(int(f[k - 1]), int(f[k])):
                    mj = models.get(j)
                    if mj is not None:
                        offx += mj["b"][0]; offy += mj["b"][1]
            cx += (rx[k] - rx[k - 1]) - offx
            cy += (ry[k] - ry[k - 1]) - offy
            xs[i + k], ys[i + k] = cx, cy
        i += len(f)
    t["x"], t["y"] = xs, ys
    if not log:
        return t
    rows = [dict(frame=fr, n=m["n"], b=float(np.hypot(*m["b"])),
                 struct_rms=m["srms"], null=m["null"], applied=m["applied"])
            for fr, m in sorted(models.items())]
    return t, pd.DataFrame(rows)


def convection_report(traj, min_n=12, n_perm=24):
    """Per-run convection summary from the significance-gated affine fit."""
    models = frame_models(traj, min_n=min_n, n_perm=n_perm)
    log = pd.DataFrame([dict(struct_rms=m["srms"], null=m["null"],
                             applied=m["applied"]) for m in models.values()])
    return dict(
        n_frames=int(len(log)),
        frac_applied=float(log["applied"].mean()),
        struct_rms_med=float(log["struct_rms"].median(skipna=True)),
        struct_rms_p90=float(log["struct_rms"].quantile(0.90)),
        excess_over_null=float((log["struct_rms"] - log["null"]).median(skipna=True)),
        verdict=("convection" if log["applied"].mean() > 0.2 else "no-convection"))


# ----- synthetic validation: removes convection, doesn't bias clean data ---

def _inject_roll(traj, omega):
    """Add a synthetic position-dependent convection (a rotational roll of rate
    omega [rad/frame] about the field centre, cumulative in frame index) to the
    raw coordinates. Global de-drift CANNOT remove this; affine should."""
    t = traj.sort_values(["particle", "frame"]).copy()
    src_x = t["x_raw"] if "x_raw" in t else t["x"]
    src_y = t["y_raw"] if "y_raw" in t else t["y"]
    cx, cy = float(src_x.mean()), float(src_y.mean())
    dtf = (t["frame"] - t["frame"].min()).to_numpy(float)
    x, y = src_x.to_numpy(float), src_y.to_numpy(float)
    nx = x + dtf * (-omega * (y - cy))
    ny = y + dtf * (omega * (x - cx))
    t["x_raw"], t["y_raw"], t["x"], t["y"] = nx, ny, nx, ny
    return t


def validate_synthetic(traj, mpp, dt, omega=0.0008, min_len=80):
    """Compare D recovered by GLOBAL vs LOCAL de-drift on (a) clean data and
    (b) data with an injected convection roll. Returns a summary dict."""
    from . import msd as msdmod, track as trackmod

    counts = traj.groupby("particle")["frame"].count()
    pids = counts[counts >= min_len].index
    base = traj[traj["particle"].isin(pids)].copy()
    if "x_raw" in base:                              # start everyone from RAW so the
        base["x"], base["y"] = base["x_raw"], base["y_raw"]   # comparison is apples-to-apples

    def med_D(tr):
        Ds = []
        for _, g in tr.groupby("particle"):
            g = g.sort_values("frame")
            lag, m, npv = msdmod.per_bead_msd(g["frame"].values, g["x"].values,
                                              g["y"].values, 40)
            fit = msdmod.fit_D(lag, m, npv, mpp, dt, 30)
            if fit and np.isfinite(fit["D_um2_s"]):
                Ds.append(fit["D_um2_s"])
        return float(np.median(Ds)), len(Ds)

    def global_dd(tr):
        drift, _ = trackmod.robust_drift(tr)
        return trackmod.subtract_drift(tr, drift)

    D_clean, n = med_D(global_dd(base))                     # MY global de-drift = reference
    D_clean_loc, _ = med_D(local_dedrift(base))             # null guard: should match
    inj = _inject_roll(base, omega)
    D_inj_glob, _ = med_D(global_dd(inj))
    D_inj_loc, _ = med_D(local_dedrift(inj))
    return dict(n_beads=n, omega=omega, D_clean=D_clean,
                D_clean_local=D_clean_loc, D_inj_global=D_inj_glob,
                D_inj_local=D_inj_loc,
                inflation_global=D_inj_glob / D_clean,
                inflation_local=D_inj_loc / D_clean,
                null_guard_ratio=D_clean_loc / D_clean)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convection de-drift synthetic test.")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--mpp", type=float, default=0.14381)
    ap.add_argument("--fps", type=float, default=9.30)
    ap.add_argument("--omega", type=float, default=0.0008)
    args = ap.parse_args()
    tr = pd.read_csv(args.traj)
    r = validate_synthetic(tr, args.mpp, 1.0 / args.fps, omega=args.omega)
    print(f"[dedrift] n_beads={r['n_beads']}  omega={r['omega']}")
    print(f"  D_clean              = {r['D_clean']:.4f} um^2/s")
    print(f"  D_clean (local dd)   = {r['D_clean_local']:.4f}  "
          f"(null-guard ratio {r['null_guard_ratio']:.3f}; want ~1.000)")
    print(f"  D_inj  (GLOBAL dd)   = {r['D_inj_global']:.4f}  "
          f"(inflation {r['inflation_global']:.2f}x  <- convection bias)")
    print(f"  D_inj  (LOCAL  dd)   = {r['D_inj_local']:.4f}  "
          f"(inflation {r['inflation_local']:.2f}x  <- should be ~1)")
