# Final report — Brownian motion (with DLA-concentration extension)

Entry point for all **final-report** deliverables and content (parallel to
`../midterm report/`). Authors: **Shaked & Nir**.

**Report split:** Shaked — Introduction / theoretical background + Methods /
experimental procedure (for the DLA extension). Nir — Results + plots (Shaked
polishes the figures). Discussion — shared.

This folder is a **curated home for the deliverables**. The live analysis
workspace (where the scripts actually run, against the raw per-week data/videos)
is [`../experiments/dla-concentration-final/`](../experiments/dla-concentration-final/).

## Contents

```
final report/
├── figures/               report-ready figures (PNG + vector PDF)
│   ├── crops/             the 7 grounded video-frame crops (inputs to the composite)
│   └── diagnostics/       method-development / verification figures
├── scripts/               analysis + figure scripts (see "Running" below)
├── docs/                  method & literature notes
└── data/                  the reliable fractal-dimension values
```

## Figures (report deliverables)

| file | what it is | status |
|---|---|---|
| `fill_fraction_vs_conc` | occupancy φ = M/πR² vs concentration (sanity check) | ⚠ φ not independent of D — reframe pending |
| `D_vs_concentration` | **effective box-counting D vs concentration**, reliable (focused) runs only; defocused greyed | ✅ (defocused slot pending) |
| `D_vs_concentration_with_crops` | the D plot + 4-over-3 gallery of the grounded frames | ✅ (defocused greyed) |
| `growth_rate_vs_conc` | late-time front speed dR/dt vs concentration | ✅ |
| `R_and_dRdt_grid` | per-concentration R(t) + dR/dt, all 7 runs + overlay | ✅ |
| `screening_masks` / `screening_crops` | open-vs-dense explainer (why D > DLA 1.71); two versions | ✅ |
| `diagnostics/*` | box-count plateaus, mask-fix probe, single-run extraction | verification only |

## The reliable result (`data/fractalD_reliable.csv`)

Effective box-counting fractal dimension, **focused runs only** (window-stability
verified, ±~0.05):

| conc | D |
|---|---|
| 0.02 % | 1.69 |
| 0.04 % | 1.91 |
| 0.06 % | 1.87 |
| 0.15 % | 1.87 |

Defocused runs (0.30 / 0.45 / 0.56 %) — being recovered separately (Fable); see
the fractal-D notes. 0.02 % is an open DLA-like fractal (~1.7); 0.04–0.15 % are
compact / effectively space-filling (D → 2, read as ~1.9 at finite size). Driver
is **concentration**, not voltage. Full reasoning + citations in
[`docs/NOTES_fractal_dimension_concentration.md`](docs/NOTES_fractal_dimension_concentration.md).

## Running the scripts

The scripts are kept here as a record. They **execute from their original
location**, `../experiments/dla-concentration-final/scripts/`, where their
relative paths to the per-week `data/` and the raw videos resolve. Figures 1–5
are CSV-only; the crop/screening figures need the raw videos (out of git) via
`WEEK4_VIDEO_DIR` / `WEEK5_VIDEO_DIR` + `ffmpeg`.

## Open items

- **Defocused runs** (0.30 / 0.45 / 0.56 %) — recovery in progress; fold into the
  D figures when ready.
- **φ figure** — decide reframe (consistency check) vs. supplement (independent
  mass-vs-concentration sanity).
- **Report prose** — Intro / theoretical background + Methods (Shaked); Methods
  gaps (apparatus/procedure specifics) still to be pinned before drafting.
- **Report source** (`.lyx` / PDF) — to be added here, mirroring `../midterm report/`.
