"""
curate.py  (pipeline)
---------------------
Track-level singleton curation -- the decisive stage. Purity over recall: we do
not need every bead, we need an UNBIASED clean subset, so we reject anything that
isn't a confident single sphere.

Per-track evidence (median over shape measured at a fixed set of GLOBAL frames,
via shape.measure_shape): roundness (resid, ring_cv), elongation (ecc), the
direct doublet test (n_cores), fit inlier fraction, and frame-to-frame radius
scatter (R_cv = focus stability). Cross-track: the rigid-pair doublet test (two
tracks that keep ~constant separation AND take near-identical steps = one rigid
body) and a mis-link test (gross step outliers = bead swaps).

NOTE on track count: the field is closed (~56 beads visible at once) but a clip
yields many more TRACKS, because buoyant polyethylene beads drift up THROUGH the
thin focal plane -- each bead is in focus only ~10 s, so tracks are short and
turn over. The longest tracks are therefore SUSPECT (a bead that never leaves
focus isn't rising freely -> likely wall-pinned). So min_len is kept modest, not
high.

Perf: shape is measured at ~n_global frames read ONCE each (not per-bead frames,
which re-read ~all frames); the flat-field is cached; rigid-pairs are KDTree
pre-filtered to spatially-near tracks; the per-bead groupby is computed once.

Emits curation.csv (every track + scores + proposed keep/reject + reason) and
curation_proposed.csv. The human confirms via the contact sheet into labels.csv.
"""

import os
import numpy as np
import pandas as pd

from . import shape

GATES = dict(min_len=60, resid=0.10, ring_cv=0.10, ecc=0.45, inlier=0.60,
             rcv=0.25, frac_multicore=0.30)
RIGID = dict(max_sep_factor=2.5, sep_cv=0.15, step_corr=0.70, min_overlap=40)
MISLINK_K = 8.0


def kept_pids(out):
    """Particle ids of the confirmed clean singles: labels.csv (keep==1, or the
    old 'type' format) if the human has confirmed, else the auto proposal."""
    lab = os.path.join(out, "labels.csv")
    if os.path.exists(lab):
        t = pd.read_csv(lab)
        if "keep" in t:
            return set(t.loc[t["keep"] == 1, "particle"].astype(int))
        if "type" in t:                                # old labels.csv format
            ok = t["type"].isin(["perfect", "singlet", "single"])
            return set(t.loc[ok, "particle"].astype(int))
    prop = os.path.join(out, "curation_proposed.csv")
    if os.path.exists(prop):
        return set(pd.read_csv(prop)["particle"].astype(int))
    return None


def measure_tracks(video, flat, groups, frame_groups, n_global=80, half=None):
    """Median shape metrics per track. Shape is sampled at n_global frames read
    ONCE each; each bead contributes the global frames where it was detected."""
    import cv2
    fr_keys = sorted(frame_groups)
    gframes = np.unique(np.linspace(fr_keys[0], fr_keys[-1],
                                    min(n_global, len(fr_keys))).astype(int))
    acc = {pid: {k: [] for k in ("R", "resid", "ring_cv", "ecc", "ncore", "inlier")}
           for pid in groups}
    cap = cv2.VideoCapture(video)
    for f in gframes:
        sub = frame_groups.get(int(f))
        if sub is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        img = fr[..., :3].mean(-1).astype(np.float32)
        if flat is not None:
            img = img - flat
        for row in sub.itertuples(index=False):
            pid = int(row.particle)
            m = shape.measure_shape(img, float(row.x_raw), float(row.y_raw),
                                    float(row.r_est), int(row.polarity), half=half)
            if not np.isfinite(m["R"]):
                continue
            a = acc[pid]
            a["R"].append(m["R"]); a["resid"].append(m["circ_resid_frac"])
            a["ring_cv"].append(m["ring_cv"]); a["ecc"].append(m["ecc"])
            a["ncore"].append(m["n_cores"]); a["inlier"].append(m["inlier_frac"])
    cap.release()

    rows = {}
    for pid, a in acc.items():
        if len(a["R"]) < 3:
            rows[pid] = None
            continue
        R = np.array(a["R"], float)
        nc = np.array(a["ncore"], float)
        rows[pid] = dict(
            R_px_med=float(np.median(R)),
            R_cv=float(R.std() / R.mean()) if R.mean() else np.nan,
            resid_med=float(np.nanmedian(a["resid"])),
            ring_cv_med=float(np.nanmedian(a["ring_cv"])),
            ecc_med=float(np.nanmedian(a["ecc"])),
            n_cores_med=float(np.median(nc)),
            frac_multicore=float(np.mean(nc >= 2)),
            inlier_med=float(np.nanmedian(a["inlier"])),
            n_shape=int(len(R)))
    return rows


def rigid_pairs(groups, agg, rigid=RIGID):
    """Resolved-doublet partners: coexisting tracks with ~constant separation AND
    correlated step vectors. KDTree-prefiltered to spatially-near tracks."""
    from scipy.spatial import cKDTree
    pids = [p for p in groups if agg.get(p) is not None]
    if len(pids) < 2:
        return {}
    pos = {p: groups[p].set_index("frame")[["x", "y"]] for p in pids}
    med = np.array([[groups[p]["x"].median(), groups[p]["y"].median()] for p in pids])
    tree = cKDTree(med)
    flagged = {}
    for i, a in enumerate(pids):
        Ra = agg[a]["R_px_med"]
        cutoff = rigid["max_sep_factor"] * 2 * Ra + 40
        for j in tree.query_ball_point(med[i], cutoff):
            if j <= i:
                continue
            b = pids[j]
            common = pos[a].index.intersection(pos[b].index)
            if len(common) < rigid["min_overlap"]:
                continue
            pa = pos[a].loc[common].to_numpy()
            pb = pos[b].loc[common].to_numpy()
            sep = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])
            sep_mean = float(sep.mean())
            if sep_mean > rigid["max_sep_factor"] * (Ra + agg[b]["R_px_med"]):
                continue
            sep_cv = float(sep.std() / sep_mean) if sep_mean else 1.0
            sa, sb = np.diff(pa, axis=0), np.diff(pb, axis=0)
            denom = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
            corr = float((sa * sb).sum() / denom) if denom > 0 else 0.0
            if sep_cv < rigid["sep_cv"] and corr > rigid["step_corr"]:
                flagged[a] = (int(b), sep_cv, corr)
                flagged[b] = (int(a), sep_cv, corr)
    return flagged


def mislink_flags(groups, k=MISLINK_K):
    """Per-track gross step outliers (bead swaps): max_step / median_step."""
    out = {}
    for pid, g in groups.items():
        s = np.hypot(np.diff(g["x"].to_numpy()), np.diff(g["y"].to_numpy()))
        if len(s) < 5:
            out[pid] = (np.nan, np.nan, False)
            continue
        med = float(np.median(s)) + 1e-9
        mx = float(s.max())
        out[pid] = (med, mx, mx > k * med)
    return out


def classify(row, gates=GATES):
    """Reasons a track is NOT a clean free single. Empty -> proposed single."""
    reason = []
    if row["n_frames"] < gates["min_len"]:
        reason.append("short")
    if row.get("rigid_partner", -1) >= 0:
        reason.append("rigid-doublet")
    # A real doublet has two cores AND an asymmetric/elongated outline. Concentric
    # diffraction rings on a (defocused) SINGLE also read as 2 cores but stay
    # round+symmetric -> require asymmetry corroboration so we don't reject them.
    asym = (row["ecc_med"] > 0.35) or (row["ring_cv_med"] > 0.08)
    if (row["n_cores_med"] >= 2 or row["frac_multicore"] > gates["frac_multicore"]) and asym:
        reason.append("two-cores")
    if row["resid_med"] > gates["resid"] or row["ring_cv_med"] > gates["ring_cv"]:
        reason.append("not-round")
    if row["ecc_med"] > gates["ecc"]:
        reason.append("elongated")
    if row["inlier_med"] < gates["inlier"]:
        reason.append("poor-fit")
    if row["R_cv"] > gates["rcv"]:
        reason.append("defocus")
    if row.get("mislink", False):
        reason.append("mislink")
    return ";".join(reason)


def run(stem, videos_dir=None, n_global=80, n_flat=60, max_frames=None,
        gates=GATES):
    from . import paths, frames as fr

    out = paths.out_dir(stem)
    tcsv = os.path.join(out, "trajectory.csv")
    if not os.path.exists(tcsv):
        raise SystemExit(f"no trajectory.csv in {out} -- run track first")
    traj = pd.read_csv(tcsv).sort_values(["particle", "frame"]).reset_index(drop=True)
    vid = paths.video(paths.video_for_run(stem), videos_dir)
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"),
                       n_sample=n_flat, max_frames=max_frames)

    groups = {int(p): g for p, g in traj.groupby("particle")}
    frame_groups = {int(f): g for f, g in traj.groupby("frame")}
    counts = {p: len(g) for p, g in groups.items()}
    print(f"[curate] {stem}: {len(groups)} tracks; measuring shape at "
          f"{n_global} global frames...")
    agg = measure_tracks(vid, flat, groups, frame_groups, n_global=n_global)
    rigid = rigid_pairs(groups, agg)
    mis = mislink_flags(groups)

    rows = []
    for pid, g in groups.items():
        a = agg.get(pid)
        if a is None:
            continue
        med_step, max_step, mlk = mis[pid]
        r = dict(particle=int(pid), n_frames=int(counts[pid]), **a)
        r["x_med"] = float(g["x_raw"].median())
        r["y_med"] = float(g["y_raw"].median())
        r["sym_med"] = float(g["sym"].median())
        r["polarity"] = int(g["polarity"].median())
        r["rigid_partner"] = int(rigid[pid][0]) if pid in rigid else -1
        r["rigid_sep_cv"] = float(rigid[pid][1]) if pid in rigid else np.nan
        r["rigid_corr"] = float(rigid[pid][2]) if pid in rigid else np.nan
        r["med_step"] = med_step
        r["max_step"] = max_step
        r["mislink"] = bool(mlk)
        r["reason"] = classify(r, gates)
        r["proposed"] = "single" if r["reason"] == "" else "reject"
        rows.append(r)

    df = pd.DataFrame(rows).sort_values(["proposed", "sym_med"],
                                        ascending=[True, False])
    df.to_csv(os.path.join(out, "curation.csv"), index=False)
    keep = df[df["proposed"] == "single"]
    keep.to_csv(os.path.join(out, "curation_proposed.csv"), index=False)

    n_single = len(keep)
    print(f"[curate] {len(df)} tracks -> PROPOSED {n_single} clean singles, "
          f"{len(df) - n_single} rejected")
    if len(df) - n_single:
        rc = df[df["proposed"] == "reject"]["reason"].str.split(";").explode()
        print("[curate] reject reasons: " +
              ", ".join(f"{k}={v}" for k, v in rc.value_counts().items()))
    print(f"[curate] wrote curation.csv + curation_proposed.csv -> {out}")
    return df


if __name__ == "__main__":   # python -m pipeline.curate run3 [--min-len 60]
    import argparse
    ap = argparse.ArgumentParser(description="Track-level singleton curation.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--n-global", type=int, default=80)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()
    g = dict(GATES)
    if args.min_len is not None:
        g["min_len"] = args.min_len
    run(args.run, n_global=args.n_global, gates=g)
