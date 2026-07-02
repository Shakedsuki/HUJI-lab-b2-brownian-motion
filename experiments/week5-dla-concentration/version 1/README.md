# Week 5 — DLA vs CuSO₄ concentration — version 1

Analysis of the three week-5 deposition videos (in
`../../week-5-last-measurements-with-shlomo/`: `run1_0.02con.mov`,
`run 2 conc 0.04.mov`, `run3_0.06C.mov`; 1280×720 @ 59.94 fps, sampled at 1 fps).
Supersedes the draft in `week-5-last-measurements-with-shlomo/analysis/`.

Deliverables (Shlomo's three requests):

1. **גרף רדיוס המעגל החוסם כתלות בזמן** → `figures/radius_vs_time_all.png`,
   per-second data in `data/radius_<run>.csv`
2. **אוברליי של העיגול על הוידאו** → `overlays/overlay_<run>.mp4` (×30 timelapse)
3. **הממד הפרקטלי לכל ריכוז** → `figures/fractalD_<run>.png`,
   `data/fractal_dimensions.json`

## Results

| run | conc | β (R ∝ (t−t₀)^β) | D (box counting) | D (growth M∝Rg^D) | R_final |
|---|---|---|---|---|---|
| run 1 | 0.02 | 0.38 ± 0.01 | **1.61 ± 0.02** | (1.33 — unreliable, sparse anisotropic) | ≥8.1 mm (top edge) |
| run 2 | 0.04 | 0.50 ± 0.01 | **1.84 ± 0.01** | 1.86 ± 0.01 | ≥9.9 mm (clipped) |
| run 3 | 0.06 | 0.49 ± 0.01 | **1.88 ± 0.01** | (1.97 — biased by central drop) | ≥10.4 mm (clipped) |

- β ≈ 0.4–0.5, near the DLA law R ∝ t^(1/D) (week 3: β ≈ 0.52).
- **D rises with concentration** 1.61 → 1.84 → 1.88: sparse DLA fan at 0.02
  (consistent with week-3's 1.65 ± 0.04 and 2-D DLA 1.71) crossing over to
  dense-branching morphology (effective D → 2) as higher concentration raises
  the ionic current and adds drift. Read 1.84/1.88 as effective dimensions
  over one decade of scales (s ≈ 4–55 px), possibly biased slightly upward by
  the 720p resolution; the independent growth mass–radius exponent for run 2
  agrees to ±0.02.
- Quoted errors: fit statistics ⊕ segmentation-threshold scan. Run-to-run
  variation (n = 1 per concentration) is NOT included and likely dominates.

## Method (v2 pipeline — `scripts/week5_analysis.py`)

Segmentation per frame (constants carried over from week-3
`growth_kinetics.py`): flat-field hysteresis threshold AND
darkened-since-first-frame, minus the HSV-masked green cathode wire,
despeckled, gated to the seed-connected component.

Three measures fixed after inspecting v1 artefacts:

1. **Registration** — every frame is aligned to the reference by phase
   correlation before differencing. (A camera bump at t ≈ 305–320 s in run 1
   shifted the frame a few px; the graph-paper grid then failed the
   "darkened" test, connected to the cluster through the wire shadow, and
   spiked the bounding circle.)
2. **Temporal persistence** — a pixel counts only if deposit-dark in the
   current frame AND ≥1 of the two previous frames (kills 1-frame transients).
3. **Monotone accumulation** — the cluster mask is a running union: once a
   pixel joins the seed-connected deposit it never leaves (a deposit cannot
   shrink). Branch dropouts — e.g. run 1's upper branch losing its root
   behind the blurred wire at t ≈ 355–410 s — no longer dent R(t).
   **Used for R(t) only**: box counting runs on the registered single final
   frame, because unioning threshold jitter over ~700 frames fattens branches
   and inflates D (measured: run 1 1.61 → 1.83 on the accumulated mask).

R(t) = radius of the minimum enclosing circle (convex hull →
cv2.minEnclosingCircle) of the accumulated cluster. mm scale from the 1 mm
graph paper at the left edge: 35.6 ± 0.2 px/mm in all three runs
(`figures/calibration_<run>.png`; the ~1.5 mm wire width confirms the grid
period). Frames where the deposit touches the frame border are flagged
(`clipped`; × in the plot, excluded from fits — the circle is then a lower
bound). Fractal D: box counting N(s) ∝ s^(−D), fit window s = 4 px … R/8,
threshold-scan systematic included.

## Caveats

- n = 1 per concentration; the D-vs-concentration trend rests on three runs.
- Runs 2–3 out-grow the left frame edge in the final third; run 1's top
  branch exits the top edge from t ≈ 415 s.
- Dense-run D at 720p is an effective (possibly slightly inflated) exponent;
  the NEF stills (DSC_0071/0072) at 6× resolution are the definitive check.
- Concentration units as labelled on the bottles; not independently verified.

## Reproducing

Extract frames once (`ffmpeg -skip_frame nokey -i <video> -vf fps=1` →
`/tmp/w5/<run>/{ref,f}` + 12 `ref` frames), then run stages in order:
`seed` → `measure` (chunked, ascending k; state checkpoints in `/tmp/w5`) →
`merge` → `figs` → `overlay` → `encode` → `fractal`.
`scripts/runner.py` automates the chunked measurement.
