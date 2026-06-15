#!/usr/bin/env python3
"""Fractal dimension of Cu electrodeposits (DLA experiment, ~12 V, CuSO4 0.29%).

Pipeline per image:
  1. Segment the dendrite: flat-field the grayscale (divide by a heavy
     Gaussian blur) to kill the illumination gradient, threshold dark pixels,
     and mask the saturated green/yellow electrode wire (HSV).
  2. Drop speckle components whose centroid falls outside the aggregate disc
     (centroid + 99th-percentile radius of *all* foreground pixels).
  3. Box-counting dimension: N(s) ~ s^-D over a fit window between the branch
     width and the finite-size saturation scale.
  4. Mass-radius dimension: M(<r) ~ r^D about the aggregate centroid, with
     each annulus mass corrected for the area occluded by the wire.

Outputs per image: a diagnostics figure (segmentation + both log-log fits)
in ../figures/, and a summary line on stdout.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"

IMAGES = ["IMG_4125_closeup.jpeg", "IMG_4127_closeup.jpeg"]

THRESHOLD = 0.85       # on the flat-fielded grayscale (background ~ 1.0)
WIRE_DILATE_PX = 25    # safety margin around the wire mask
MIN_SPECKLE_PX = 5     # components smaller than this are JPEG/paper noise


# ---------------------------------------------------------------- segmentation

def segment(path):
    """Return (dendrite mask, wire mask, centre (x, y), radius R) in px."""
    img = cv2.imread(str(path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # flat-field: local background from a blur much wider than any branch
    bg = cv2.GaussianBlur(gray, (0, 0), 101)
    flat = gray / (bg + 1e-6)

    h, s, _ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wire = ((s > 60) & (h > 20) & (h < 90)).astype(np.uint8)
    wire = cv2.dilate(wire, np.ones((WIRE_DILATE_PX, WIRE_DILATE_PX), np.uint8))

    dend = ((flat < THRESHOLD) & (wire == 0)).astype(np.uint8)

    # aggregate disc from ALL foreground pixels (the deposit dominates the
    # foreground, so centroid/radius are robust to sparse paper speckle)
    ys, xs = np.nonzero(dend)
    cx, cy = xs.mean(), ys.mean()
    R = np.percentile(np.hypot(xs - cx, ys - cy), 99)

    # drop tiny components and anything clearly outside the aggregate disc
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dend)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_SPECKLE_PX:
            continue
        keep[i] = np.hypot(cent[i, 0] - cx, cent[i, 1] - cy) <= 1.1 * R
    dend = keep[lab].astype(np.uint8)

    # recentre on the cleaned mask
    ys, xs = np.nonzero(dend)
    cx, cy = xs.mean(), ys.mean()
    R = np.percentile(np.hypot(xs - cx, ys - cy), 99)
    return img, dend, wire, (cx, cy), R


# ---------------------------------------------------------------- box counting

def box_count(binary, sizes):
    H, W = binary.shape
    counts = []
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        blocks = binary[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3))
        counts.append(blocks.sum())
    return np.array(counts)


def fit_loglog(x, y):
    """Least-squares slope of log y vs log x with its standard error."""
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    dof = max(len(lx) - 2, 1)
    resid_var = (res[0] / dof) if len(res) else 0.0
    cov = resid_var * np.linalg.inv(A.T @ A)
    return coef[0], np.sqrt(cov[0, 0]), coef[1]


def boxcount_dimension(dend, R):
    ys, xs = np.nonzero(dend)
    crop = dend[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sizes = np.unique(np.round(2 ** np.arange(1, np.log2(min(crop.shape) / 2), 0.25)).astype(int))
    N = box_count(crop, sizes)
    ok = N > 0
    sizes, N = sizes[ok], N[ok]

    # fit window: above the branch width (~8 px at this resolution, where the
    # local slope stops being dominated by the solid branch interior) and
    # below R/8 where finite size pushes the slope to 2
    lo, hi = 8, R / 8
    win = (sizes >= lo) & (sizes <= hi)
    D, dD, b = fit_loglog(sizes[win], N[win])
    return -D, dD, sizes, N, win, b


# ----------------------------------------------------------------- mass-radius

def mass_radius_dimension(dend, wire, centre, R):
    cx, cy = centre
    H, W = dend.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy)

    # the method assumes an unobstructed centre: if the wire covers much of
    # the inner disc the small-r mass is unmeasurable and the fit, left with
    # only the dense outer rim, is biased high -> flag as unreliable
    inner = r < 0.25 * R
    inner_visible = (wire[inner] == 0).mean()
    reliable = inner_visible > 0.75

    edges = np.geomspace(4, R, 40)
    mass, valid = [], []
    visible = wire == 0
    for r0, r1 in zip(edges[:-1], edges[1:]):
        ann = (r >= r0) & (r < r1)
        vis = (ann & visible).sum()
        tot = ann.sum()
        if tot == 0 or vis / tot < 0.4:       # annulus mostly occluded: skip
            mass.append(0.0)
            valid.append(False)
            continue
        # correct the annulus mass for the wire-occluded fraction
        mass.append(dend[ann].sum() * tot / vis)
        valid.append(True)
    mass = np.cumsum(mass)
    rmid = np.sqrt(edges[:-1] * edges[1:])
    valid = np.array(valid) & (mass > 0)

    # fit window: outside the electrode blob, inside 0.8 R (rim is undergrown)
    win = valid & (rmid >= 30) & (rmid <= 0.8 * R)
    D, dD, b = fit_loglog(rmid[win], mass[win])
    return D, dD, rmid, mass, win, b, reliable


# ------------------------------------------------------------------ diagnostics

def diagnostics(name, img, dend, wire, centre, R, bc, mr):
    Dbc, dDbc, sizes, N, winb, bb = bc
    Dmr, dDmr, rmid, mass, winm, bm, mr_ok = mr

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    vis[dend > 0] = (220, 30, 30)
    vis[wire > 0] = (30, 90, 220)
    axes[0].imshow(vis)
    th = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(centre[0] + R * np.cos(th), centre[1] + R * np.sin(th), "y--", lw=1)
    axes[0].set_title(f"{name}\nsegmentation (red = deposit, blue = wire mask)")
    axes[0].axis("off")

    axes[1].loglog(sizes, N, "o", ms=4, color="gray", label="all scales")
    axes[1].loglog(sizes[winb], N[winb], "o", ms=5, color="C3", label="fit window")
    ss = np.array([sizes[winb].min(), sizes[winb].max()])
    axes[1].loglog(ss, np.exp(bb) * ss ** (-Dbc), "k-", lw=1)
    axes[1].set_xlabel("box size s (px)")
    axes[1].set_ylabel("N(s)")
    axes[1].set_title(f"box counting: D = {Dbc:.3f} ± {dDbc:.3f}")
    axes[1].legend()

    axes[2].loglog(rmid, np.where(mass > 0, mass, np.nan), "o", ms=4, color="gray")
    axes[2].loglog(rmid[winm], mass[winm], "o", ms=5, color="C0", label="fit window")
    rr = np.array([rmid[winm].min(), rmid[winm].max()])
    axes[2].loglog(rr, np.exp(bm) * rr ** Dmr, "k-", lw=1)
    axes[2].set_xlabel("radius r (px)")
    axes[2].set_ylabel("M(<r) (px, occlusion-corrected)")
    flag = "" if mr_ok else "  [unreliable: occluded centre]"
    axes[2].set_title(f"mass-radius: D = {Dmr:.3f} ± {dDmr:.3f}{flag}")
    axes[2].legend()

    fig.tight_layout()
    out = FIGS / f"fractal_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    results = []
    for fname in IMAGES:
        name = Path(fname).stem
        img, dend, wire, centre, R = segment(DATA / fname)
        bc = boxcount_dimension(dend, R)
        mr = mass_radius_dimension(dend, wire, centre, R)
        out = diagnostics(name, img, dend, wire, centre, R, bc, mr)
        mr_ok = mr[6]
        print(f"{name}:  R = {R:.0f} px,  fg = {dend.sum()} px")
        print(f"  box counting  D = {bc[0]:.3f} ± {bc[1]:.3f}")
        print(f"  mass-radius   D = {mr[0]:.3f} ± {mr[1]:.3f}"
              + ("" if mr_ok else "  [excluded: wire occludes centre]"))
        print(f"  -> {out.relative_to(HERE.parent)}")
        results.append(bc[0])
        if mr_ok:
            results.append(mr[0])

    allD = np.array(results)
    # ±0.03 systematic from the segmentation threshold (0.85 ± 0.05 scan)
    syst = 0.03
    err = np.sqrt(allD.std(ddof=1) ** 2 + syst ** 2)
    print(f"\ncombined: D = {allD.mean():.2f} ± {err:.2f}"
          f"  (over {len(allD)} estimates, incl. ±{syst} threshold syst;"
          f" DLA theory: 1.71)")


if __name__ == "__main__":
    main()
