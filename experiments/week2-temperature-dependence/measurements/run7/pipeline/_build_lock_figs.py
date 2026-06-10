"""
Build two WEEK1-style figures for run7 (T=16.8 C).

FIG1 lock_fig1_msd.png : time-averaged MSD <r^2> vs lag time tau for 3
  representative CLEAN-FIT beads spanning size (smallest / median / largest r
  among the 15), with the short-lag linear fit <r^2> = 4 D tau + c (fit_lag=20).

FIG2 lock_fig2_kb.png  : D vs 1/r for all 15 clean-fit beads + the red
  Stokes-Einstein line through origin using the WEEK1 median(D*r) slope.

Run from .../week2-temperature-dependence:  python measurements/run7/pipeline/_build_lock_figs.py
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import physics, msd
try:
    from pipeline import figstyle
    figstyle.set_style()
except Exception:
    figstyle = None

# ----- constants / calibration ------------------------------------------------
KB        = 1.380649e-23                      # accepted Boltzmann [J/K]
MPP       = 0.14381                           # um/px
FPS       = 9.299
DT        = 1.0 / FPS
T_C       = 16.8
T_K       = T_C + 273.15
ETA       = physics.water_viscosity_Pa_s(T_C) # Pa.s  (1.086 cP)
ETA_CP    = ETA * 1e3
FIT_LAG   = 20

HERE = os.path.dirname(os.path.abspath(__file__))            # .../measurements/run7/pipeline
KB_CSV   = os.path.join(HERE, "kb_per_bead.csv")
TRAJ_CSV = os.path.join(HERE, "trajectory.csv")
FIG1 = os.path.join(HERE, "lock_fig1_msd.png")
FIG2 = os.path.join(HERE, "lock_fig2_kb.png")

# ----- clean-fit set ----------------------------------------------------------
df = pd.read_csv(KB_CSV)
clean = df[(df.r_um >= 0.5) & (df.alpha >= 0.9) & (df.alpha <= 1.15)
           & (df.n_frames >= 200) & (df.intercept_um2.abs() < 0.2)].copy()
clean = clean.sort_values("r_um").reset_index(drop=True)
assert len(clean) == 15, f"expected 15 clean beads, got {len(clean)}"

# WEEK1 median(D*r) slope -> k_B
slope_um = float(np.median(clean.D_um2_s * clean.r_um))      # um^3/s  ( = D*r )
# k_B = 6 pi eta / T * slope, with slope in SI [m^3/s]
slope_SI = slope_um * 1e-18                                  # um^3/s -> m^3/s (um^2 * um = 1e-12*1e-6)
kB_est = 6.0 * np.pi * ETA / T_K * slope_SI
kB_med_perbead = float(clean.kb_i.median())
print(f"[chk] median(D*r) = {slope_um:.5f} um^3/s")
print(f"[chk] k_B (slope) = {kB_est:.4e} = {kB_est/KB:.3f}x accepted")
print(f"[chk] k_B (median per-bead kb_i) = {kB_med_perbead:.4e} = {kB_med_perbead/KB:.3f}x")

# ----- 3 representative beads spanning size -----------------------------------
i_small = 0
i_large = len(clean) - 1
i_med   = (len(clean) - 1) // 2                              # 7 -> the median-r bead
reps = {"smallest": clean.iloc[i_small], "median": clean.iloc[i_med],
        "largest": clean.iloc[i_large]}
for k, row in reps.items():
    print(f"[rep] {k:8s} particle={int(row.particle)}  r={row.r_um:.4f} um  "
          f"D(csv)={row.D_um2_s:.4f} um^2/s")

# ----- trajectory (already de-drifted: x,y vs raw x_raw,y_raw) ----------------
traj = pd.read_csv(TRAJ_CSV)

def bead_msd_fit(pid):
    g = traj[traj.particle == pid].sort_values("frame")
    lag, msd_px2, npair = msd.per_bead_msd(g.frame.values, g.x.values,
                                           g.y.values, max_lag=2 * FIT_LAG + 5)
    fit = msd.fit_D(lag, msd_px2, npair, MPP, DT, FIT_LAG)
    tau = lag * DT
    msd_um2 = msd_px2 * MPP * MPP
    return tau, msd_um2, fit

# ============================ FIG 1 ===========================================
fig1, ax = plt.subplots(figsize=(7.2, 5.2))
colors = {"smallest": "C0", "median": "C1", "largest": "C2"}
fig1_beads = {}
for k, row in reps.items():
    pid = int(row.particle)
    tau, msd_um2, fit = bead_msd_fit(pid)
    D = fit["D_um2_s"]; c = fit["intercept_um2"]
    fig1_beads[k] = dict(particle=pid, r_um=float(row.r_um), D=D)
    col = colors[k]
    ax.plot(tau, msd_um2, "o", color=col, ms=4.5, alpha=0.85,
            label=f"p{pid}: r={row.r_um:.2f} $\\mu$m, D={D:.3f} $\\mu$m$^2$/s")
    tg = np.linspace(0, FIT_LAG * DT, 50)
    ax.plot(tg, 4.0 * D * tg + c, "-", color=col, lw=2.0, alpha=0.95)
    print(f"[fig1] p{pid}: D={D:.4f} um^2/s  c={c:.4f} um^2  alpha={fit['alpha']:.3f}")

ax.set_xlim(0, FIT_LAG * DT)
ax.set_ylim(0, None)
ax.set_xlabel(r"lag time $\tau$ [s]")
ax.set_ylabel(r"MSD $\langle r^2 \rangle$  [$\mu$m$^2$]")
ax.set_title("run7: MSD linear in time")
ax.legend(loc="upper left", fontsize=9, title="clean-fit beads (4D$\\tau$+c fit)")
fig1.tight_layout()
if figstyle is not None:
    figstyle.save(fig1, FIG1)
else:
    fig1.savefig(FIG1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"[fig1] saved -> {FIG1}")

# ============================ FIG 2 ===========================================
inv_r = 1.0 / clean.r_um.values
D_vals = clean.D_um2_s.values

fig2, ax2 = plt.subplots(figsize=(7.2, 5.2))
ax2.scatter(inv_r, D_vals, s=46, color="C0", edgecolor="0.25", lw=0.6, zorder=3,
            label="clean-fit beads (n=15)")
# red Stokes-Einstein line through origin, slope = median(D*r) = slope_um (um^3/s)
xr = np.linspace(0, inv_r.max() * 1.08, 50)
ax2.plot(xr, slope_um * xr, "r-", lw=2.2, zorder=2,
         label=r"Stokes-Einstein: $D=\frac{k_B T}{6\pi\eta}\,\frac{1}{r}$")
ax2.set_xlim(0, inv_r.max() * 1.08)
ax2.set_ylim(0, max(D_vals.max(), (slope_um * xr).max()) * 1.08)
ax2.set_xlabel(r"$1/r$  [1/$\mu$m]")
ax2.set_ylabel(r"$D$  [$\mu$m$^2$/s]")
ax2.set_title("run7: Stokes-Einstein  $D \\propto 1/r$")
txt = (f"$k_B$ = ({kB_est*1e23:.2f}) e-23 = {kB_est/KB:.2f}x accepted\n"
       f"n=15, T=16.8C, $\\eta$=1.086cP")
ax2.text(0.04, 0.96, txt, transform=ax2.transAxes, va="top", ha="left",
         fontsize=10, bbox=dict(boxstyle="round,pad=0.45", fc="white",
                                ec="0.4", alpha=0.92))
ax2.legend(loc="lower right", fontsize=9)
fig2.tight_layout()
if figstyle is not None:
    figstyle.save(fig2, FIG2)
else:
    fig2.savefig(FIG2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[fig2] saved -> {FIG2}")
print(f"[fig2] kB box value xacc = {kB_est/KB:.4f}")

# summary line for the caller
import json
print("RESULT " + json.dumps(dict(
    fig1=FIG1, fig2=FIG2, kB_est=kB_est, kB_over_acc=kB_est/KB,
    kB_box_xacc=round(kB_est/KB, 2),
    fig1_beads=fig1_beads)))
