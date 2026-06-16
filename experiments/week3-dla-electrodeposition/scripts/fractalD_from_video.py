#!/usr/bin/env python3
"""Standalone, readable mass-radius fractal-dimension plot from a kinetics CSV.

The mass-radius dimension measured from a single growing deposit: as the cluster
grows, plot M (deposit area) against its radius of gyration Rg over the
self-similar growth window; the log-log slope is the fractal dimension D.

Usage:  python3 scripts/fractalD_from_video.py [clip_tag]   (default run4_dense)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"
import importlib.util
spec = importlib.util.spec_from_file_location("gk", HERE / "growth_kinetics.py")
gk = importlib.util.module_from_spec(spec); spec.loader.exec_module(gk)

tag = sys.argv[1] if len(sys.argv) > 1 else "run4_dense"
a = np.genfromtxt(DATA / f"kinetics_{tag}.csv", delimiter=",", names=True)
t, M, R95, Rg = a["t_s"], a["M_px"], a["R95_px"], a["Rg_px"]

t0 = gk.nucleation_time(t, M)
w = gk.growth_window(t, M, Rg)
k = gk.fit_kinetics(t, M, R95, Rg, w, t0)
s = gk.window_systematic(t, M, R95, Rg, t0)
D, dD = k["D"], float(np.hypot(k["dD"], s["D_sys"]))
good = M > 0

fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))

# main: M vs Rg log-log with the fit
ax[0].loglog(Rg[good], M[good], "o", ms=3, color="0.7", label="all frames")
ax[0].loglog(Rg[w], M[w], "o", ms=4, color="C3", label="growth-window fit")
rr = np.array([Rg[w].min(), Rg[w].max()])
b = np.exp(np.median(np.log(M[w]) - D * np.log(Rg[w])))
ax[0].loglog(rr, b * rr ** D, "k-", lw=2)
# DLA reference slope (1.71), anchored at the window midpoint
rmid = np.sqrt(rr[0] * rr[1]); mmid = b * rmid ** D
ax[0].loglog(rr, mmid * (rr / rmid) ** 1.71, "g--", lw=1.2, label="DLA slope 1.71")
ax[0].set_xlabel("radius of gyration  Rg  [px]")
ax[0].set_ylabel("mass  M  [deposit px]")
ax[0].set_title(f"{tag}: mass-radius fractal dimension")
ax[0].legend(loc="upper left", fontsize=9)
ax[0].text(0.97, 0.05, f"D = {D:.2f} ± {dD:.2f}\n(static stills: 1.65 ± 0.04)",
           transform=ax[0].transAxes, ha="right", va="bottom", fontsize=11,
           bbox=dict(boxstyle="round", fc="white", ec="C3"))

# local slope dlogM/dlogRg vs Rg (curvature / scale-free range)
o = np.argsort(Rg[good]); rg = Rg[good][o]; mm = M[good][o]
lr, lm = np.log(rg), np.log(mm)
kk = 30
loc = np.full(len(lr), np.nan)
for i in range(len(lr)):
    lo, hi = max(0, i - kk), min(len(lr), i + kk + 1)
    if hi - lo > 8:
        loc[i] = np.polyfit(lr[lo:hi], lm[lo:hi], 1)[0]
ax[1].semilogx(rg, loc, "-", color="C0")
ax[1].axhline(D, color="k", ls=":", label=f"fit D = {D:.2f}")
ax[1].axhline(1.71, color="g", ls="--", label="DLA 1.71")
ax[1].axhline(2.0, color="0.6", ls=":", label="solid (D=2)")
ax[1].axvspan(Rg[w].min(), Rg[w].max(), color="C3", alpha=0.12, label="fit window")
ax[1].set_ylim(1.0, 2.2)
ax[1].set_xlabel("radius of gyration  Rg  [px]")
ax[1].set_ylabel("local slope  dlogM/dlogRg")
ax[1].set_title("local mass-radius slope (scale-free check)")
ax[1].legend(fontsize=8, loc="lower right")

fig.suptitle(f"Fractal dimension from the {tag} recording  (M ∝ Rg^D)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = FIGS / f"fractalD_{tag}.png"
fig.savefig(out, dpi=140); plt.close(fig)
print(f"D({tag}) = {D:.2f} +/- {dD:.2f}   n_fit={k['n_fit']}   -> {out}")
