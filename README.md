# HUJI Lab B2 — Brownian Motion

**Course:** Lab B2, 2026
**Students:** Shaked Sukiennik, Nir Cohen
**Instructor:** Shlomo Winberg

---

## Overview

Experimental study of the Brownian motion of micron-scale polymer microspheres
in water, tracked via optical microscopy. The experiment measures the mean
squared displacement (MSD) versus time, particle size, and temperature, and uses
the Stokes-Einstein relation to extract Boltzmann's constant.

**Core relation:**

```
⟨r²⟩ = (2k_BT / 3πηa) · t  =  4Dt        (2D projection)
```

where `D = k_BT / 6πηa` is the diffusion coefficient (Stokes-Einstein).

---

## Repo layout

```
brownian-motion/
├── docs/                                # colloquium tracker, reference notes
├── experiments/
│   └── week1-system-calibration/
│       ├── scripts/                     # the analysis pipeline (see below)
│       ├── calibration/                 # µm/px scale (scale.json) + check image
│       ├── measurements/<run>/          # trajectory/msd/radius/labels CSVs
│       │   └── figures/                 # that run's plot1.png, plot2.png
│       ├── figures/                     # cross-run aggregate
│       ├── runs.json, videos_meta.json  # per-run physics + acquisition metadata
│       └── videos/                      # raw .avi (gitignored)
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

## Setup

```bash
pip install -r requirements.txt
```

---

## Key references

1. Jia et al., *Am. J. Phys.* **75**, 111 (2007) — primary technical reference
2. Andelman & Diamant, *Tehuda* **26**(3) (2007) — background reading (Hebrew)
3. Lab booklet: *חוברת מעבדה תנועה בראונית 2023*
