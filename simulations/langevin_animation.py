"""
simulations/langevin_animation.py
==================================
HUJI Lab B2 — Brownian Motion, 2026
Shaked Sukiennik & Nir Cohen

Animated simulation of a single Brownian particle obeying the Langevin equation:

    m dv/dt = -γv + F(t)        [Eq. 1, Jia et al. 2007]
    γ = 6πηa                    [Eq. 2, Stokes drag]

INTEGRATOR — overdamped limit
------------------------------
For a microsphere in water the momentum relaxation time is:

    τ_relax = m/γ  ~  30 μs

which is orders of magnitude below any experimentally observable timescale
(camera frame rate ~ 10–30 Hz). We therefore integrate the overdamped limit:

    γ dx/dt = F(t)   →   Δx = √(2D·dt) · ξ,   ξ ~ N(0,1)

Display forces are reconstructed from the displacement each step:
    v_eff   = Δx / dt              (effective velocity)
    F_rand  =  γ · v_eff           (random force that drove the step)
    F_drag  = −γ · v_eff           (Stokes drag, exactly opposes F_rand)

Panels  (left → right)
-----------------------
1. Particle frame  — 2D view that follows the particle; shows live force vectors
2. Lab frame       — fixed 3D cube; full trajectory builds up from the origin
3. MSD             — ⟨r²⟩ vs t compared to theoretical 4Dt

Usage
-----
    python simulations/langevin_animation.py

    # optional flags:
    python simulations/langevin_animation.py --backend Qt5Agg --steps 6000 --fps 60 --seed 7
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Langevin / Stokes drag animator")
parser.add_argument("--backend", default="TkAgg", help="Matplotlib backend")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--steps",   type=int, default=4000, help="Simulation steps")
parser.add_argument("--fps",     type=int, default=60)
args, _ = parser.parse_known_args()

matplotlib.use(args.backend)

# ── Physical parameters ───────────────────────────────────────────────────────
kT    = 1.0
eta   = 1.0
a     = 0.5
gamma = 6 * np.pi * eta * a          # γ = 6πηa  ≈ 9.42
D     = kT / gamma                   # D = k_BT / γ
m     = 0.3                          # used only for display labels
dt    = 0.05
N     = args.steps
TRAIL = 120
tau_relax = m / gamma

print(f"  γ = 6πηa       = {gamma:.4f}")
print(f"  D = k_BT/γ     = {D:.4f}")
print(f"  τ_relax = m/γ  = {tau_relax:.4f}   (overdamped: dt/τ = {dt/tau_relax:.0f})")

# ── Simulate — overdamped Langevin ────────────────────────────────────────────
rng      = np.random.default_rng(args.seed)
pos      = np.zeros((N, 2))
vel      = np.zeros((N, 2))
f_rnd    = np.zeros((N, 2))
f_drg    = np.zeros((N, 2))
sigma_x  = np.sqrt(2 * D * dt)

for i in range(1, N):
    xi       = rng.standard_normal(2)
    dx       = sigma_x * xi
    pos[i]   = pos[i-1] + dx
    vel[i]   = dx / dt
    f_rnd[i] =  gamma * vel[i]
    f_drg[i] = -gamma * vel[i]

r2    = np.sum(pos**2, axis=1)
times = np.arange(N) * dt

# 3-D cube half-size: cover ~3σ of expected displacement
CUBE  = max(15.0, 3.0 * np.sqrt(4 * D * N * dt))

# ── Style ─────────────────────────────────────────────────────────────────────
BG, PANEL   = '#0d1117', '#161b22'
WHITE, GREY = '#e6edf3', '#8b949e'
EDGE_COL    = '#30363d'
C_VEL, C_DRG, C_RND = '#ffd700', '#f85149', '#3fb950'

# ── Figure — 3 panels ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 6), facecolor=BG)
fig.suptitle(
    r'Langevin: $m\,\dot{v}=-\gamma v+F(t)$   '
    r'Stokes: $\gamma=6\pi\eta a$   '
    r'[overdamped, $\Delta x=\sqrt{2D\,dt}\,\xi$]',
    color=WHITE, fontsize=11, y=0.995
)

gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.15, 0.9], wspace=0.35)
ax1  = fig.add_subplot(gs[0])                  # 2D particle frame
ax3d = fig.add_subplot(gs[1], projection='3d') # 3D lab frame
ax2  = fig.add_subplot(gs[2])                  # MSD

# ── Common 2D axis styling ────────────────────────────────────────────────────
for ax in (ax1, ax2):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(EDGE_COL)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)

# ── 3D axis styling ───────────────────────────────────────────────────────────
ax3d.set_facecolor(BG)
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill      = True
    pane.set_facecolor(PANEL)
    pane.set_edgecolor(EDGE_COL)
ax3d.tick_params(colors=GREY, labelsize=7, pad=1)
ax3d.xaxis.label.set_color(GREY)
ax3d.yaxis.label.set_color(GREY)
ax3d.zaxis.label.set_color(GREY)
for line in ax3d.xaxis.get_gridlines() + ax3d.yaxis.get_gridlines() + ax3d.zaxis.get_gridlines():
    line.set_color(EDGE_COL)
    line.set_alpha(0.4)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — particle frame (2D, follows particle)
# ══════════════════════════════════════════════════════════════════════════════
WIN = 10
ax1.set_xlim(-WIN, WIN)
ax1.set_ylim(-WIN, WIN)
ax1.set_aspect('equal')
ax1.set_title('Particle frame  (2D, follows particle)', color=WHITE, fontsize=9, pad=6)
ax1.set_xlabel('x  [a.u.]')
ax1.set_ylabel('y  [a.u.]')

trail1,  = ax1.plot([], [], '-', color='#58a6ff', alpha=0.3, lw=1.0, zorder=2)
dot1,    = ax1.plot([], [], 'o', color=WHITE, ms=8, zorder=5, mec='#58a6ff', mew=1.5)

ax1.legend(handles=[
    mpatches.Patch(color=C_VEL, label=r'velocity $v_\mathrm{eff}$'),
    mpatches.Patch(color=C_DRG, label=r'Stokes drag $-\gamma v$'),
    mpatches.Patch(color=C_RND, label=r'random $F(t)$'),
], loc='upper left', facecolor=BG, edgecolor=EDGE_COL,
   labelcolor=WHITE, fontsize=7.5, framealpha=0.9)

ax1.text(
    0.99, 0.03,
    f'$\\gamma={gamma:.2f}$,  $D={D:.3f}$\n'
    f'$\\tau_{{relax}}={tau_relax:.3f}$,  Re$\\sim10^{{-6}}$',
    transform=ax1.transAxes, color=GREY, fontsize=7.5,
    ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor=BG, alpha=0.85, edgecolor=EDGE_COL)
)
time_txt = ax1.text(0.5, 0.97, '', transform=ax1.transAxes,
                    color=GREY, fontsize=8, ha='center', va='top')

_arrows = []   # live arrow artists in ax1

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — lab frame (3D fixed cube)
# ══════════════════════════════════════════════════════════════════════════════
ax3d.set_xlim(-CUBE, CUBE)
ax3d.set_ylim(-CUBE, CUBE)
ax3d.set_zlim(-CUBE, CUBE)
ax3d.set_xlabel('x', labelpad=2)
ax3d.set_ylabel('y', labelpad=2)
ax3d.set_zlabel('z', labelpad=2)
ax3d.set_title('Lab frame  (3D, fixed)', color=WHITE, fontsize=9, pad=4)
ax3d.view_init(elev=22, azim=45)

# Cube wireframe
def _draw_cube(ax, s, col=EDGE_COL, alpha=0.45, lw=0.8):
    verts = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [ 1,-1,-1],[ 1,-1,1],[ 1,1,-1],[ 1,1,1]], dtype=float) * s
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
             (3,7),(4,5),(4,6),(5,7),(6,7)]
    for a, b in edges:
        ax.plot3D(*zip(verts[a], verts[b]), color=col, lw=lw, alpha=alpha)

_draw_cube(ax3d, CUBE)

# Floor grid — shows the xy plane the particle actually moves in
_gx = np.linspace(-CUBE, CUBE, 7)
for gv in _gx:
    ax3d.plot([gv, gv], [-CUBE, CUBE], [0, 0], color=EDGE_COL, lw=0.5, alpha=0.3)
    ax3d.plot([-CUBE, CUBE], [gv, gv], [0, 0], color=EDGE_COL, lw=0.5, alpha=0.3)

# Origin marker
ax3d.plot([0], [0], [0], 'o', color=GREY, ms=4, alpha=0.6)

# 3D trajectory artists
trail3d, = ax3d.plot([], [], [], '-', color='#58a6ff', alpha=0.5, lw=1.2, zorder=3)
dot3d,   = ax3d.plot([], [], [], 'o', color=WHITE, ms=7, zorder=5,
                      mec='#58a6ff', mew=1.2)
# Vertical "shadow" drop-line from particle down to z = -CUBE
shadow_v, = ax3d.plot([], [], [], '--', color=GREY, alpha=0.25, lw=0.8)
# XY shadow projected onto floor (z = -CUBE offset so it sits just below)
shadow_xy, = ax3d.plot([], [], [], '-', color='#58a6ff', alpha=0.12, lw=0.8)

ax3d.text2D(0.02, 0.96, 'particle confined to z = 0 plane',
            transform=ax3d.transAxes, color=GREY, fontsize=7, va='top')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — MSD
# ══════════════════════════════════════════════════════════════════════════════
ax2.set_title(r'$\langle r^2 \rangle$ vs time', color=WHITE, fontsize=9, pad=6)
ax2.set_xlabel('t  [a.u.]')
ax2.set_ylabel(r'$\langle r^2 \rangle$  [a.u.]')
ax2.set_xlim(0, N * dt)
ax2.set_ylim(0, 6 * D * N * dt)

t_th = np.linspace(0, N * dt, 400)
ax2.plot(t_th, 4 * D * t_th, '--', color='#f0883e', lw=1.8, alpha=0.7,
         label=r'$4Dt$  (theory)')
msd_line, = ax2.plot([], [], '-', color='#58a6ff', lw=1.4, label='simulation')
ax2.legend(facecolor=BG, edgecolor=EDGE_COL, labelcolor=WHITE, fontsize=8)

# ── Arrow helper (panel 1 only) ───────────────────────────────────────────────
VEL_SC, FORCE_SC = 1.5, 0.12

def _arrow(ax, x, y, dx, dy, color, lw=2.1, hw=0.20):
    if np.hypot(dx, dy) < 1e-6:
        return None
    return ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
        arrowprops=dict(arrowstyle=f'->,head_width={hw},head_length={hw*0.8}',
                        color=color, lw=lw), zorder=6)

# ── Animation ─────────────────────────────────────────────────────────────────
def init():
    trail1.set_data([], [])
    dot1.set_data([], [])
    trail3d.set_data([], [])
    trail3d.set_3d_properties([])
    dot3d.set_data([], [])
    dot3d.set_3d_properties([])
    shadow_v.set_data([], [])
    shadow_v.set_3d_properties([])
    shadow_xy.set_data([], [])
    shadow_xy.set_3d_properties([])
    msd_line.set_data([], [])
    return trail1, dot1, trail3d, dot3d, shadow_v, shadow_xy, msd_line

def update(frame):
    global _arrows
    i = frame
    x, y = pos[i]

    # ── panel 1: particle frame ───────────────────────────────────────────────
    s = max(0, i - TRAIL)
    trail1.set_data(pos[s:i+1, 0], pos[s:i+1, 1])
    dot1.set_data([x], [y])
    ax1.set_xlim(x - WIN, x + WIN)
    ax1.set_ylim(y - WIN, y + WIN)

    for arr in _arrows:
        arr.remove()
    _arrows.clear()
    for dx, dy, col in [
        (*vel[i]   * VEL_SC,   C_VEL),
        (*f_drg[i] * FORCE_SC, C_DRG),
        (*f_rnd[i] * FORCE_SC, C_RND),
    ]:
        a = _arrow(ax1, x, y, dx, dy, col)
        if a is not None:
            _arrows.append(a)

    time_txt.set_text(f't = {times[i]:.1f}')

    # ── panel 2: lab frame 3D ─────────────────────────────────────────────────
    # Full trail from origin (lab frame — never pans)
    trail3d.set_data(pos[:i+1, 0], pos[:i+1, 1])
    trail3d.set_3d_properties(np.zeros(i+1))        # z = 0 plane

    dot3d.set_data([x], [y])
    dot3d.set_3d_properties([0])

    # Vertical drop-line from particle to bottom of cube
    shadow_v.set_data([x, x], [y, y])
    shadow_v.set_3d_properties([0, -CUBE])

    # XY floor shadow
    shadow_xy.set_data(pos[:i+1, 0], pos[:i+1, 1])
    shadow_xy.set_3d_properties(np.full(i+1, -CUBE))

    # ── panel 3: MSD ──────────────────────────────────────────────────────────
    msd_line.set_data(times[:i+1], r2[:i+1])

    return trail1, dot1, trail3d, dot3d, shadow_v, shadow_xy, msd_line, time_txt

ani = animation.FuncAnimation(
    fig, update,
    frames=range(1, N, 2),
    init_func=init,
    interval=max(1, 1000 // args.fps),
    blit=False
)

# tight_layout with 3D axes requires rect to avoid suptitle clash;
# use subplots_adjust as the reliable fallback
fig.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.10, wspace=0.35)
plt.show()
