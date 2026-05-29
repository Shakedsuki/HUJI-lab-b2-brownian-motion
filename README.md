# HUJI Lab B2 — Brownian Motion

**Course:** Lab B2, 2026
**Students:** Shaked Sukiennik, Nir Cohen
**Instructor:** Shlomo Winberg

---

## Overview

Experimental study of the Brownian motion of polyethylene microspheres in water
and water/glycerol mixtures, tracked via optical microscopy. The experiment
measures the mean squared displacement (MSD) as a function of time, particle
size, viscosity, and temperature, and uses the Stokes-Einstein relation to
extract Boltzmann's constant.

**Core relation:**

```
⟨r²⟩ = (2k_BT / 3πηa) · t  =  4Dt        (2D projection)
```

where `D = k_BT / 6πηa` is the diffusion coefficient (Stokes-Einstein).

---

## Repo layout

```
brownian-motion/
├── docs/                       # Reference material, lab notes
├── simulations/                # Physics simulations & visualisations
│   ├── langevin_animation.py   # Animated Langevin + Stokes drag demo
│   └── __init__.py
├── requirements.txt
└── README.md
```

> **data/** and **analysis/** directories will be added once measurements begin.

---

## Simulations

### `simulations/langevin_animation.py`

Real-time 2D animation of a single Brownian particle governed by the Langevin
equation:

```
m dv/dt  =  -γv  +  F(t)        (Langevin, Eq. 1 — Jia et al. 2007)
γ        =  6πηa                 (Stokes drag, Eq. 2)
```

**What it shows:**

- Live particle trajectory with fading trail
- Three force vectors updating each frame:
  - 🟡 velocity **v**
  - 🔴 Stokes drag **−γv** (always opposite to velocity)
  - 🟢 random thermal force **F(t)**
- MSD vs time panel, compared to theoretical `4Dt` (orange dashed)

**Run:**

```bash
python simulations/langevin_animation.py

# optional flags
python simulations/langevin_animation.py --backend Qt5Agg --steps 6000 --fps 60
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Key references

1. Jia et al., *Am. J. Phys.* **75**, 111 (2007) — primary technical reference
2. Andelman & Diamant, *Tehuda* **26**(3) (2007) — background reading (Hebrew)
3. Lab booklet: *חוברת מעבדה תנועה בראונית 2023*
