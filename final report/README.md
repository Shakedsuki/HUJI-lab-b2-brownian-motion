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
│   ├── crops/             the 6 grounded video-frame crops (inputs to the composite)
│   └── diagnostics/       method-development / verification figures
├── scripts/               analysis + figure scripts (see "Running" below)
├── docs/                  method & literature notes
└── data/                  the reliable fractal-dimension values
```

**Dataset note:** the 0.30 % run is **excluded** (defocused video; <0.6 decades
of scaling range at every moment of the run; all salvage routes failed their
validation controls — see `docs/NOTES_defocused_runs_recovery.md`). The other
two formerly-"defocused" runs were recovered: 0.56 % was never actually
defocused, 0.45 %'s mild blur is handled by a validated window rule +
deconvolution. All figures below are built from the remaining six runs.

## Figures (report deliverables)

| file | what it is | status |
|---|---|---|
| `fill_fraction_vs_conc` | occupancy φ = M/πR² vs concentration (sanity check) | ⚠ φ not independent of D — reframe pending |
| `D_vs_concentration` | **effective box-counting D vs concentration**, all six reliable runs; 0.30 % exclusion annotated | ✅ publication-ready |
| `D_vs_concentration_with_crops` | the D plot + 3-over-3 gallery of the grounded frames, each captioned with its D | ✅ publication-ready |
| `growth_rate_vs_conc` | late-time front speed dR/dt vs concentration (6 runs) | ✅ |
| `R_and_dRdt_grid` | per-concentration R(t) + dR/dt, 6 runs + overlay | ✅ |
| `screening_masks` / `screening_crops` | open-vs-dense explainer (why D > DLA 1.71); two versions | ✅ |
| `diagnostics/*` | box-count plateaus, mask-fix probe, single-run extraction | verification only |

## The reliable result (`data/fractalD_reliable.csv`)

Effective box-counting fractal dimension (faithful mask, window
[max(branch width, 3σ_blur), R/3], window-stability verified):

| conc | D | note |
|---|---|---|
| 0.02 % | 1.69 ± 0.05 | open DLA-like fractal |
| 0.04 % | 1.91 ± 0.05 | compact |
| 0.06 % | 1.87 ± 0.04 | compact (cleanest plateau) |
| 0.15 % | 1.87 ± 0.06 | compact (validation anchor) |
| 0.30 % | — excluded | defocused; no exponent quotable (compact, D → 2) |
| 0.45 % | 1.93 ± 0.07 | compact; recovered (validated stack + RL) |
| 0.56 % | 1.86 ± 0.06 | compact; recovered (was never defocused) |

0.02 % is an open DLA-like fractal (~1.7); all denser runs are compact /
effectively space-filling (D → 2, read as ~1.9 at finite size — do not rank
values within the dense band). Driver is **concentration**, not voltage. Full
reasoning + citations in
[`docs/NOTES_fractal_dimension_concentration.md`](docs/NOTES_fractal_dimension_concentration.md)
(§1a covers why the recovered values exceeding DLA's 1.71 is expected); the
recovery/exclusion audit is
[`docs/NOTES_defocused_runs_recovery.md`](docs/NOTES_defocused_runs_recovery.md).

## Running the scripts

The scripts are kept here as a record. They **execute from their original
location**, `../experiments/dla-concentration-final/scripts/`, where their
relative paths to the per-week `data/` and the raw videos resolve. Figures 1–5
are CSV-only; the crop/screening figures need the raw videos (out of git) via
`WEEK4_VIDEO_DIR` / `WEEK5_VIDEO_DIR` + `ffmpeg`.

## Open items

- **φ figure** — decide reframe (consistency check) vs. supplement (independent
  mass-vs-concentration sanity).
- **Report prose** — Intro / theoretical background + Methods (Shaked); Methods
  gaps (apparatus/procedure specifics) still to be pinned before drafting.
- **Report source** (`.lyx` / PDF) — to be added here, mirroring `../midterm report/`.
