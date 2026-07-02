# Week 4 — DLA vs CuSO₄ concentration (0.15 / 0.30 / 0.45 / 0.56) — version 1

Same pipeline and deliverables as `week5-dla-concentration/version 1`
(registration + temporal persistence + monotone seed-connected mask;
minimum-enclosing-circle R(t); ×30 overlay; box-counting D), applied to the
four week-4 videos in `../week4-dla-no-shlomo/` (1280×720 @ 59.94 fps,
sampled at 1 fps). Calibration from the 1 mm graph paper:
**49.7–50.2 px/mm** in all four runs (±0.5 %, camera fixed).

## Run-quality context (important)

A desk lamp was added mid-session for better lighting; it unintentionally
**heated the cell**, discolouring the deposit where the light fell (bleached
copper-brown/yellow vs black). Per run:

| run | conc | lamp | focus | quality notes |
|---|---|---|---|---|
| 1 | 0.56 | OFF until ~min 6 (t≈369 s), then ON | ok | dim early (deposit detected only from ~58 s); lamp-on + camera re-frame at 369–388 s; heated region bleached |
| 2 | 0.45 | ON from start | **out of focus** | warm cast (white-balanced calibration); continuous small camera drift |
| 3 | 0.30 | no lamp | not focused | two large camera re-framings (t≈196 s, ≈306 s) |
| 4 | 0.15 | no lamp | focused | **best measurement** — zero camera motion, clean throughout |

The unnamed `DSC_00xx.mov` clips in the raw folder are continuations of these
runs after stopping to add the lamp / re-focus; they are **not** included here
(no continuous time axis across the stop).

## Pipeline additions forced by the above (vs week 5)

1. **Wire-mask hue band 25–95** (was 18–95): the copper-brown deposit of the
   high-concentration runs falls at H ≈ 10–20 and was being masked out.
2. **White-balanced, detrended grid calibration**: run 2's warm cast and the
   wire crossing the grid strip otherwise break the autocorrelation.
3. **Camera-motion guard**: per-frame registration shift is logged
   (`dx_px, dy_px`); a border band ~|shift| wide is excluded (warp smear), and
   frames with shift > 60 px or an anomalous mask burst (> 30 k new px) are
   flagged `moved` — the cluster is **frozen**, not contaminated. This handles
   the mid-run re-framings; measurement resumes automatically when the view
   returns (it genuinely does in run 1 at t ≈ 390–405 s).
4. **Reference-free final-frame mask for fractal D** (flat-field hysteresis −
   wire − blue grid, disc-gated): final frames sit after re-framings, so the
   deposit-free start reference no longer applies.

## Results

| run | conc | β (R ∝ (t−t₀)^β) | D (box counting) | R_final |
|---|---|---|---|---|
| 1 | 0.56 | 0.36 ± 0.01 | 1.90 ± 0.01 | ≥11.5 mm |
| 2 | 0.45 | 0.40 ± 0.01 | 1.91 ± 0.01 | ≥9.2 mm |
| 3 | 0.30 | 0.30 ± 0.01 | 1.88 ± 0.01 | ≥10.0 mm |
| 4 | 0.15 | 0.38 ± 0.01 | **1.85 ± 0.01** | 7.2 mm |

(β statistical errors only; week-5 analysis showed a ±0.05–0.1 fit-window
systematic. The growth mass–radius cross-check is **not reliable** in week 4
— diffuse halo + near-compact deposits — and is reported only for
completeness in `data/fractal_dimensions.json`.)

**Combined with week 5** (`figures/D_vs_concentration_w4w5.png`): D rises
from 1.6 at 0.02 through ≈1.85 and saturates near the compact limit D → 2
across 0.15–0.56 — all four week-4 concentrations sit deep in the
dense-branching regime, consistent with the morphology transition seen at
0.02 → 0.06. Note week-4 D values carry extra systematic risk: defocus
(runs 2–3) blurs inter-branch gaps and biases box counting upward, and run 1's
lamp heating adds an uncontrolled temperature covariate.

## Deliverables

1. R(t): `figures/radius_vs_time_all.png`, `data/radius_run*.csv`
   (`moved` column = camera-motion-censored frames, excluded from fits)
2. Overlays: `overlays/overlay_run*.mp4` (×30 timelapse)
3. Fractal D: `figures/fractalD_run*.png`, `data/fractal_dimensions.json`,
   combined trend `figures/D_vs_concentration_w4w5.png`

## Caveats

- n = 1 per concentration; lamp/focus/temperature vary **between** runs, so
  the flat D(c) trend within week 4 should not be over-read.
- run 1 pre-lamp segment (t < 369 s) is the only heating-free part of that run;
  the fit window ends well before it.
- The continuation clips (DSC_00xx) could extend R(t) per run if stitched with
  per-segment references — not attempted here.
