"""
plot1_report.py  (week1-system-calibration)
-------------------------------------------
The reportable version of Plot 1 (booklet Part 1): a handful of CLEAN beads'
mean-squared displacement vs lag time, showing directly that

        <r^2>(tau) = 4 D tau + c        (2D projection => factor 4)

is LINEAR in tau (normal diffusion), with per-bead D from the short-lag slope.
Two panels: linear axes (the straight-line "money" plot) + log-log (slope ~ 1).

Beads are auto-picked from msd.csv as the cleanest available (long, in-focus,
round, near-zero intercept) and chosen to SPAN a range of D so the differing
slopes preview the size dependence (Plot 2). Override with --beads.

This recomputes the time-averaged MSD directly from trajectory.csv (no trackpy
dependency) so it is self-contained.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/plot1_report.py run3 --tag d21m600
    python scripts/plot1_report.py run3                       # untagged dir
    python scripts/plot1_report.py run3 --beads 28 24 297     # explicit beads
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths
import figure_style


def tamsd(sub, mpp, dt, max_lag):
    """Time-averaged MSD (um^2) vs lag time (s) for one bead, gap-aware."""
    sub = sub.sort_values("frame")
    f = sub["frame"].values.astype(int)
    x = sub["x"].values * mpp
    y = sub["y"].values * mpp
    pos = {fr: (xx, yy) for fr, xx, yy in zip(f, x, y)}
    lags, msd = [], []
    for lag in range(1, max_lag + 1):
        sq = [(pos[fr + lag][0] - pos[fr][0]) ** 2 + (pos[fr + lag][1] - pos[fr][1]) ** 2
              for fr in f if fr + lag in pos]
        if len(sq) >= 5:
            lags.append(lag * dt)
            msd.append(float(np.mean(sq)))
    return np.array(lags), np.array(msd)


def pick_clean(cdir, n):
    """Auto-pick n clean SINGLE spheres spanning the radius range.

    Restricts to human-labelled singles (labels.csv perfect/singlet) when
    available -- the trustworthy, appearance-based curation that Plot 2 uses --
    then applies only size-UNCONFOUNDED quality gates (track length, focus
    stability, fit intercept). The old ecc gate is intentionally dropped: small
    dim singles read high apparent eccentricity from intensity noise, so an ecc
    cut wrongly rejects good small beads and favours the large low-ecc doublets
    (this is why Plot 1 used to feature a bead Plot 2 calls a doublet). Beads are
    spread across RADIUS so their differing slopes preview Plot 2; returned small
    -> large.
    """
    m = pd.read_csv(os.path.join(cdir, "msd.csv"))
    rpath = os.path.join(cdir, "radius.csv")
    lpath = os.path.join(cdir, "labels.csv")
    if os.path.exists(rpath):
        rad = pd.read_csv(rpath)
        rc = [c for c in ["particle", "r_um", "circ_resid_frac", "inlier_frac"] if c in rad.columns]
        m = m.merge(rad[rc], on="particle", how="left")
    if os.path.exists(lpath):
        lab = pd.read_csv(lpath)
        keep = lab[lab["type"].isin(["perfect", "singlet"])]["particle"]
        m = m[m["particle"].isin(keep)]
    elif "circ_resid_frac" in m:                    # objective single-sphere proxy
        m = m[(m["circ_resid_frac"] < 0.10) & (m.get("inlier_frac", 1) > 0.70)]
    # One permissive quality bar (track length + focus stability) builds the
    # candidate pool; we then SPAN radius across it. Gates are size-unconfounded
    # and intercept is loose: small beads carry a larger localisation offset and
    # the 4Dt+c fit absorbs it, so excluding them would strand Plot 1 at the
    # large (slow) end and hide the size dependence it exists to show. Spanning
    # by construction (not "first gate that yields n") prevents the clean LARGE
    # beads from monopolising all three slots.
    for g in [dict(nf=300, cv=0.30, ic=0.40), dict(nf=150, cv=0.45, ic=0.70)]:
        c = m[(m.n_frames >= g["nf"]) & (m.size_cv < g["cv"]) &
              (m.intercept_um2.abs() < g["ic"])]
        if len(c) >= n:
            break
    if len(c) < n:
        c = m.sort_values("n_frames", ascending=False).head(max(n, 1))
    if "r_um" in c and c["r_um"].notna().any():
        c = c.sort_values("r_um")                       # small radius first
    else:
        c = c.sort_values("D_um2_s", ascending=False)   # high D == small first
    idx = np.linspace(0, len(c) - 1, n).astype(int)
    return c.iloc[idx]["particle"].astype(int).tolist()


def main():
    ap = argparse.ArgumentParser(description="Reportable Plot 1: clean-bead MSD vs t.")
    ap.add_argument("run", help="run stem, e.g. run3")
    ap.add_argument("--tag", default=None, help="measurements/<run>/<tag>/ (match track.py)")
    ap.add_argument("--beads", type=int, nargs="*", default=None,
                    help="explicit particle ids; default = auto-pick cleanest")
    ap.add_argument("--n-show", type=int, default=3, help="how many beads to feature")
    ap.add_argument("--fit-lag", type=int, default=20, help="fit slope over lags <= this (frames)")
    ap.add_argument("--max-lag", type=int, default=55, help="display MSD up to this lag (frames)")
    args = ap.parse_args()

    stem = args.run
    cdir = _paths.clip_dir(stem)
    if args.tag:
        cdir = os.path.join(cdir, args.tag)
    tcsv = os.path.join(cdir, "trajectory.csv")
    mcsv = os.path.join(cdir, "msd.csv")
    if not os.path.exists(tcsv):
        sys.exit(f"no trajectory.csv in {cdir} -- run track.py first")

    mpp = _paths.load_scale() or 1.0
    meta_video = _paths.load_runs().get("runs", {}).get(stem, {}).get("video", stem + ".avi")
    fps = _paths.fps_of(meta_video) or 9.30
    dt = 1.0 / fps

    # radius lookup so curves are labelled by ACTUAL size (small=smallest r)
    r_of = {}
    rpath = os.path.join(cdir, "radius.csv")
    if os.path.exists(rpath):
        rr = pd.read_csv(rpath)
        r_of = {int(p): float(r) for p, r in zip(rr["particle"], rr["r_um"]) if pd.notna(r)}

    beads = args.beads
    if not beads:
        if not os.path.exists(mcsv):
            sys.exit("no msd.csv to auto-pick beads; run msd_fit.py or pass --beads")
        beads = pick_clean(cdir, args.n_show)
    elif r_of:
        beads = sorted(beads, key=lambda p: r_of.get(p, np.inf))   # small -> large
    print(f"[plot1] {stem}: mpp={mpp} um/px, fps={fps:.3f}; beads {beads} "
          f"(r={[round(r_of.get(b, float('nan')), 2) for b in beads]} um)")

    traj = pd.read_csv(tcsv)
    figure_style.set_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = ["C0", "C1", "C3", "C2", "C4", "C5"]
    size_words = ["small", "medium", "large"]

    def bead_label(k, pid):
        r = r_of.get(pid, np.nan)
        base = size_words[k] + " bead" if (args.n_show == 3 and k < 3) else f"bead {pid}"
        return base + (f" ($r$={r:.2f} $\\mu$m)" if np.isfinite(r) else "")

    guide = None

    for k, pid in enumerate(beads):
        sub = traj[traj["particle"] == pid]
        lags, msd = tamsd(sub, mpp, dt, args.max_lag)
        if len(lags) < 3:
            continue
        guide = lags
        fm = lags <= args.fit_lag * dt
        (sl, ic), cov = np.polyfit(lags[fm], msd[fm], 1, cov=True)
        D, Derr = sl / 4.0, float(np.sqrt(cov[0, 0])) / 4.0
        col = colors[k % len(colors)]
        lab = bead_label(k, pid)
        ax[0].plot(lags, msd, "o", ms=3, color=col, alpha=0.6)
        xs = np.linspace(0, lags[fm].max(), 50)
        ax[0].plot(xs, sl * xs + ic, "-", color=col, lw=1.8,
                   label=f"{lab}: D = {D:.3f} $\\pm$ {Derr:.4f} " + r"$\mu$m$^2$/s")
        ax[1].loglog(lags, msd, "o", ms=3, color=col, alpha=0.6, label=lab)

    ax[0].set_xlabel(r"lag time  $\tau$  [s]")
    ax[0].set_ylabel(r"$\langle r^2 \rangle$   [$\mu$m$^2$]")
    ax[0].set_title(r"MSD linear in $t$   (fit $\langle r^2\rangle = 4D\tau + c$,  "
                    rf"$\tau \leq {args.fit_lag/fps:.1f}$ s)", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_xlim(0, args.max_lag * dt)

    if guide is not None:
        gx = np.array([guide.min(), guide.max() * 0.5])
        ax[1].loglog(gx, gx / gx[0] * 0.05, "k--", lw=1, label="slope 1")
    ax[1].set_xlabel(r"lag time  $\tau$  [s]")
    ax[1].set_ylabel(r"$\langle r^2 \rangle$   [$\mu$m$^2$]")
    ax[1].set_title(r"log$-$log: slope $\approx$ 1  $\Rightarrow$  normal diffusion", fontsize=10)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    path = figure_style.savefig("plot1.png", fig=fig, outdir=os.path.join(cdir, "figures"))
    plt.close(fig)
    print(f"[plot1] wrote {path}")


if __name__ == "__main__":
    main()
