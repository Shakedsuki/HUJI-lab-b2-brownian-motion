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

Weeks 1–2 cover the Brownian-motion / k_B measurement above. **Week 3 is a
separate experiment** — diffusion-limited aggregation (DLA) studied via copper
electrodeposition (see [Week 3](#week-3--dla-by-cu-electrodeposition) below).

---

## Repo layout

```
brownian-motion/
├── docs/                                # colloquium tracker, reference notes
├── experiments/
│   ├── week1-system-calibration/        # k_B pipeline (see below)
│   │   ├── scripts/                     # the analysis pipeline
│   │   ├── calibration/                 # µm/px scale (scale.json) + check image
│   │   ├── measurements/<run>/          # trajectory/msd/radius/labels CSVs
│   │   ├── figures/                     # cross-run aggregate
│   │   ├── runs.json, videos_meta.json  # per-run physics + acquisition metadata
│   │   └── videos/                      # raw .avi (gitignored)
│   ├── week2-temperature-dependence/    # k_B vs temperature
│   └── week3-dla-electrodeposition/     # DLA: fractal dimension + growth kinetics
│       ├── scripts/                     # fractal_dimension.py, growth_kinetics.py, ...
│       ├── data/                        # analysis inputs + kinetics_<clip>.csv
│       ├── figures/                     # fractal/kinetics diagnostics
│       └── media/                       # camera stills (videos live outside the repo)
├── requirements.txt
└── README.md
```

---

## Pipeline (`experiments/week1-system-calibration/scripts/`)

```
track.py        video → trajectory.csv      (locate → link → drift-subtract)
msd_fit.py      trajectory → msd.csv        (per-bead MSD → D = slope/4)
measure_radius.py  trajectory+video → radius.csv  (outer-edge circle fit)
label_beads.py  (optional) manual single/doublet labels; else curation is objective
plot1_report.py per-run MSD-vs-t demo  →  measurements/<run>/figures/plot1.png
plot2_report.py per-run D-vs-1/r → k_B  →  measurements/<run>/figures/plot2.png
aggregate.py    pool runs → combined k_B + error budget → figures/aggregate_*.png
```

Shared helpers: `physics.py` (constants, viscosity, Stokes-Einstein),
`_paths.py` (per-week path resolver), `figure_style.py` (consistent figures).

---

## Week 3 — DLA by Cu electrodeposition

A separate experiment: copper electrodeposited in a thin quasi-2D cell (CuSO₄
0.29 %, ~12 V, central cathode wire + copper ring anode) grows a branched
dendrite by diffusion-limited aggregation. Two measurements
(`experiments/week3-dla-electrodeposition/`):

- **Fractal dimension (still photos):** `D = 1.65 ± 0.04` — box-counting +
  mass-radius on the dried deposits (2D-DLA theory ≈ 1.71).
- **Growth kinetics (videos):** tracking the deposit frame-by-frame gives
  `M ∝ Rg^D` with `D = 1.6 ± 0.1` (independently reproducing the static value),
  `M ∝ t^α` with `α ≈ 0.9` (≈ constant-current / Faradaic deposition), and the
  aggregate radius `Rg ∝ t^β` with `β ≈ 0.52 ≈ 1/D` (the DLA radius law
  `R ∝ t^(1/D)`). The three laws close self-consistently (`D ≈ α/β`).

See the [Week-3 README](experiments/week3-dla-electrodeposition/README.md) for
method, per-clip results, and figure captions.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Key references

1. Jia et al., _Am. J. Phys._ **75**, 111 (2007) — primary technical reference
2. Andelman & Diamant, _Tehuda_ **26**(3) (2007) — background reading (Hebrew)
3. Lab booklet: _חוברת מעבדה תנועה בראונית 2023_
