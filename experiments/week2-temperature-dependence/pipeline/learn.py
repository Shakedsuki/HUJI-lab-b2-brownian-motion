"""
learn.py  (pipeline)
--------------------
Supervised refinement of singleton curation from HUMAN labels. This learns "what
a clean sphere looks like" from your tags -- it does NOT touch the physics answer
(it fits morphology, statistically independent of k_B), so it doesn't violate the
no-tuning-to-target rule. Kept deliberately interpretable for the report.

Training labels (independent of the auto-proposal, to avoid circularity):
  * the OLD pipeline's labels.csv (type in {perfect,singlet}=single, else reject),
    transferred onto THIS pipeline's tracks by position (offset-corrected);
  * this pipeline's labels.csv ONLY if you actually edited it (keep != proposed
    for some rows -> human review happened).

Three refinements (all reported with honest validation):
  1. supervised_gates  : per-feature one-sided cutoff at a target precision ->
     data-derived replacement for the hand-set GATES (+ in-sample PR).
  2. classifier        : logistic regression + shallow tree on the feature
     vector, leave-one-out CV precision/recall; flags beads where it disagrees
     with the gate proposal (for your eye, not to override).
  3. appearance_prototype : mean normalized radial profile of confirmed singles;
     a 'proto_match' correlation score for every track.

Outputs: learned_gates.json, curation_learned.csv, learn_report.png.
Run AFTER you've confirmed run3 labels: python -m pipeline.learn run3
"""

import json
import os
import numpy as np
import pandas as pd

from . import paths, figstyle

# features used for the classifier; gate tuning uses the one-sided subset below
FEATURES = ["resid_med", "ring_cv_med", "ecc_med", "R_cv", "inlier_med",
            "sym_med", "frac_multicore", "n_cores_med"]
# (feature, direction): -1 = single is LOWER, +1 = single is HIGHER
GATE_DIRS = {"resid_med": -1, "ring_cv_med": -1, "ecc_med": -1, "R_cv": -1,
             "frac_multicore": -1, "n_cores_med": -1, "inlier_med": +1,
             "sym_med": +1}
SINGLE_TYPES = {"perfect", "singlet", "single"}


def build_training_set(stem, tol_px=16.0):
    """Return (X df with FEATURES, y array 1=single/0=reject, source list)."""
    from . import review
    out = paths.out_dir(stem)
    cur = pd.read_csv(os.path.join(out, "curation.csv"))

    y = pd.Series(index=cur.index, dtype="float")        # NaN = no label
    # (a) old labels transferred by position
    old_type = review.match_new_to_old(stem, cur, tol_px=tol_px)
    has_old = old_type != "unlabeled"
    y[has_old] = np.isin(old_type[has_old], list(SINGLE_TYPES)).astype(float)
    # (b) my labels.csv, but only if it was actually edited (diverges from proposal)
    lab_path = os.path.join(out, "labels.csv")
    if os.path.exists(lab_path):
        lab = pd.read_csv(lab_path)
        if "keep" in lab and "proposed" in lab:
            edited = (lab["keep"] != (lab["proposed"] == "single").astype(int)).any()
            if edited:
                kmap = dict(zip(lab["particle"], lab["keep"]))
                for i, pid in enumerate(cur["particle"]):
                    if pid in kmap:
                        y.iloc[i] = float(kmap[pid])     # human-confirmed wins
                print("[learn] using human-edited labels.csv as supervision")
    mask = y.notna()
    X = cur.loc[mask, ["particle"] + FEATURES].reset_index(drop=True)
    print(f"[learn] training set: {int(mask.sum())} labeled tracks "
          f"({int(y[mask].sum())} single / {int((y[mask] == 0).sum())} reject)")
    return X, y[mask].to_numpy(), cur


def supervised_gates(X, y, target_precision=0.95):
    """Per-feature one-sided cutoff achieving >= target precision with max recall."""
    pos = y == 1
    n_single = pos.sum()
    gates, table = {}, []
    for f, d in GATE_DIRS.items():
        if f not in X:
            continue
        v = X[f].to_numpy()
        cands = np.unique(v)
        best = None
        for t in cands:
            keep = v <= t if d < 0 else v >= t
            tp = (keep & pos).sum()
            kp = keep.sum()
            if kp == 0:
                continue
            prec = tp / kp
            rec = tp / max(n_single, 1)
            if prec >= target_precision:
                if best is None or rec > best[2]:
                    best = (t, prec, rec)
        if best is None:                                 # relax: best precision
            for t in cands:
                keep = v <= t if d < 0 else v >= t
                kp = keep.sum()
                if kp == 0:
                    continue
                prec = (keep & pos).sum() / kp
                rec = (keep & pos).sum() / max(n_single, 1)
                if best is None or prec > best[1]:
                    best = (t, prec, rec)
        gates[f] = float(best[0])
        table.append((f, d, float(best[0]), best[1], best[2]))
    return gates, table


def fit_classifier(X, y):
    """Logistic regression + shallow tree with leave-one-out CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    from sklearn.metrics import precision_score, recall_score, confusion_matrix

    Xf = X[FEATURES].to_numpy()
    out = {}
    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced",
                                                   max_iter=1000)),
        "tree": DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                       random_state=0),
    }
    loo = LeaveOneOut()
    for name, mdl in models.items():
        if len(y) >= 6 and y.sum() >= 2 and (y == 0).sum() >= 2:
            pred = cross_val_predict(mdl, Xf, y, cv=loo)
            out[name] = dict(
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                confusion=confusion_matrix(y, pred).tolist())
        else:
            out[name] = dict(precision=np.nan, recall=np.nan, confusion=None)
        mdl.fit(Xf, y)                                   # full-data fit for predict
    out["_logreg_coef"] = dict(zip(
        FEATURES, models["logreg"].named_steps["logisticregression"].coef_[0].round(3)))
    return models, out


def appearance_prototype(stem, X, y, n_frames=5, rmax_factor=2.0, n_r=40):
    """Mean normalized radial profile of confirmed singles + a match score for
    every track. Returns (rgrid, prototype, {pid: match})."""
    import cv2
    from scipy.ndimage import map_coordinates
    from . import frames as fr

    out = paths.out_dir(stem)
    cur = pd.read_csv(os.path.join(out, "curation.csv")).set_index("particle")
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    vid = paths.video(paths.video_for_run(stem))
    flat = fr.flat_field(vid, n_sample=60)
    rgrid = np.linspace(0.0, rmax_factor, n_r)
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    def profile(img, x, y0, R, pol):
        rs = np.clip(rgrid * R, 0.5, None)
        XX = x + np.cos(th)[:, None] * rs[None, :]
        YY = y0 + np.sin(th)[:, None] * rs[None, :]
        p = map_coordinates(img, [YY.ravel(), XX.ravel()], order=1,
                            mode="nearest").reshape(len(th), n_r).mean(0)
        p = (p - p[-3:].mean()) * pol                    # bg-subtract, polarity-align
        s = p.std()
        return p / s if s > 0 else p

    # gather sampled-frame profiles per track (read each frame once)
    want = {}
    for pid in cur.index:
        g = traj[traj["particle"] == pid].sort_values("frame")
        if len(g) == 0 or pid not in cur.index:
            continue
        R = cur.loc[pid, "R_px_med"]
        if not np.isfinite(R) or R <= 2:
            continue
        for j in np.linspace(0, len(g) - 1, min(n_frames, len(g))).astype(int):
            row = g.iloc[j]
            want.setdefault(int(row["frame"]), []).append(
                (int(pid), float(row["x_raw"]), float(row["y_raw"]), R,
                 int(row["polarity"])))
    profs = {}
    cap = cv2.VideoCapture(vid)
    for f in sorted(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr_ = cap.read()
        if not ok:
            continue
        img = fr_[..., :3].mean(-1).astype(np.float32) - flat
        for pid, x, y0, R, pol in want[f]:
            profs.setdefault(pid, []).append(profile(img, x, y0, R, pol))
    cap.release()
    mean_prof = {pid: np.mean(ps, axis=0) for pid, ps in profs.items() if ps}

    single_pids = set(X.loc[y == 1, "particle"].astype(int))
    proto_src = [mean_prof[p] for p in single_pids if p in mean_prof]
    if not proto_src:
        return rgrid, None, {}
    prototype = np.mean(proto_src, axis=0)
    match = {pid: float(np.corrcoef(pr, prototype)[0, 1])
             for pid, pr in mean_prof.items()}
    return rgrid, prototype, match


def run(stem, target_precision=0.95, tol_px=16.0):
    out = paths.out_dir(stem)
    X, y, cur = build_training_set(stem, tol_px=tol_px)
    if len(y) < 6 or y.sum() < 2 or (y == 0).sum() < 2:
        raise SystemExit(f"[learn] too few labeled beads ({len(y)}) -- confirm "
                         f"more in labels.csv or label another run")

    gates, table = supervised_gates(X, y, target_precision)
    print(f"[learn] supervised gates (target precision {target_precision}):")
    for f, d, t, p, r in table:
        print(f"    {f:16s} {'<=' if d < 0 else '>='} {t:7.3f}   "
              f"(in-sample P={p:.2f} R={r:.2f})")
    json.dump(gates, open(os.path.join(out, "learned_gates.json"), "w"), indent=2)

    models, clf = fit_classifier(X, y)
    print(f"[learn] classifier LOO-CV: "
          f"logreg P={clf['logreg']['precision']:.2f} R={clf['logreg']['recall']:.2f} | "
          f"tree P={clf['tree']['precision']:.2f} R={clf['tree']['recall']:.2f}")
    print(f"[learn] logreg weights: {clf['_logreg_coef']}")

    rgrid, prototype, match = appearance_prototype(stem, X, y)

    # annotate full curation with classifier prediction + prototype match + flags
    full = cur.copy()
    Xall = full[FEATURES].to_numpy()
    full["clf_logreg"] = models["logreg"].predict(Xall).astype(int)
    full["clf_tree"] = models["tree"].predict(Xall).astype(int)
    full["proto_match"] = full["particle"].map(match)
    full["gate_single"] = (full["proposed"] == "single").astype(int)
    full["clf_disagree"] = (full["clf_logreg"] != full["gate_single"]).astype(int)
    full.to_csv(os.path.join(out, "curation_learned.csv"), index=False)
    nd = int(full["clf_disagree"].sum())
    print(f"[learn] wrote curation_learned.csv; classifier disagrees with gates on "
          f"{nd} tracks (review these)")

    _report(stem, X, y, table, clf, rgrid, prototype, match, out)
    return gates, clf


def _report(stem, X, y, table, clf, rgrid, prototype, match, out):
    import matplotlib.pyplot as plt
    figstyle.set_style()
    fig = plt.figure(figsize=(14, 8))
    # per-feature single vs reject distributions with learned cut
    for i, (f, d, t, p, r) in enumerate(table[:6]):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.hist(X.loc[y == 1, f], bins=15, alpha=0.7, color="C2", label="single")
        ax.hist(X.loc[y == 0, f], bins=15, alpha=0.6, color="C3", label="reject")
        ax.axvline(t, color="k", ls="--", lw=1.2)
        ax.set_title(f"{f}\n{'<=' if d < 0 else '>='}{t:.3f} P={p:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    # prototype profile
    ax = fig.add_subplot(2, 4, 7)
    if prototype is not None:
        ax.plot(rgrid, prototype, "C0-")
        ax.axvline(1.0, color="0.5", ls=":", lw=1)
        ax.set_xlabel("r / R"); ax.set_ylabel("norm. I (polarity-aligned)")
        ax.set_title("learned bead prototype", fontsize=8)
    # proto-match separation
    ax = fig.add_subplot(2, 4, 8)
    if match:
        ms = X["particle"].map(match)
        ax.hist(ms[y == 1].dropna(), bins=15, alpha=0.7, color="C2", label="single")
        ax.hist(ms[y == 0].dropna(), bins=15, alpha=0.6, color="C3", label="reject")
        ax.set_xlabel("proto_match (corr)"); ax.set_title("prototype match", fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle(f"{stem}: supervised refinement -- gates (in-sample P/R), "
                 f"classifier LOO P={clf['logreg']['precision']:.2f}/"
                 f"R={clf['logreg']['recall']:.2f}, prototype")
    p = figstyle.save(fig, os.path.join(out, "learn_report.png"), dpi=130)
    plt.close(fig)
    print(f"[learn] wrote learn_report.png -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Supervised refinement from labels.")
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--tol-px", type=float, default=16.0)
    args = ap.parse_args()
    run(args.run, target_precision=args.target_precision, tol_px=args.tol_px)
