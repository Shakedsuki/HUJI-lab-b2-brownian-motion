# Week 5 — DLA vs CuSO₄ concentration — FINAL

Merged best-of from two **independently written analysis pipelines** of the
same three videos (`version 1/` and `version 2/`; raw footage in
`experiments/week-5-last-measurements-with-shlomo/`). The two pipelines were
cross-validated against each other before merging:

- R(t) series agree to **RMS 0.05–0.21 mm** on all trustworthy segments;
- identical frame-edge contact times (±2 s) and final radii;
- same calibration, 35.6 ± 0.2 px/mm from the 1 mm graph paper (autocorrelation
  of the grid profile; the ~1.5 mm wire width confirms the period);
- box-counting D agrees within errors (see `figures/D_vs_concentration.png`);
- refitting **both** datasets under a common fit window and t₀ gives β equal
  to ±0.01 — every apparent β disagreement between the versions is fit-window
  convention, quantified in `data/beta_window_systematics.json`.

## Results

| | run 1 — 0.02 | run 2 — 0.04 | run 3 — 0.06 |
|---|---|---|---|
| morphology | sparse DLA-like dendrite | dense radial | near-compact disc |
| β (R ∝ (t−t₀)^β) | 0.38 (0.38–0.51)* | 0.50 (0.37–0.50)* | 0.49 (0.40–0.49)*, late front ~linear (≈26 µm/s) |
| **D, box counting** | **1.62** | **1.85** | **1.84** |
| D, combined (3 frames × 3 thr × 2 estimators) | 1.54 ± 0.09 | 1.89 ± 0.05 | 1.97 ± 0.21 |
| D, independent pipeline cross-check | 1.61 ± 0.02 | 1.84 ± 0.01 | 1.88 ± 0.01† |
| first frame-edge contact | 400 s | 600 s | 417 s |
| final enclosing R (lower bound) | ≥10.3 mm | ≥13.8 mm | ≥15.4 mm |

\* parentheses: spread across fit-window/t₀ conventions — the dominant error;
statistical errors (±0.01) are not meaningful alone.
† measured on the (clipped, post-thickening) final frame; the combined value
uses the last edge-free frame, methodologically preferable for runs 2–3.

**Physics:** concentration changes the morphology class, not just the rate.
At 0.02, growth is DLA-like (D ≈ 1.6, just under the 2-D DLA 1.71, as expected
with field-driven drift; matches week-3's 1.65 ± 0.04 at 0.29 %). With rising
concentration the ionic current grows, the deposit crosses into the
dense-branching regime, D rises to ≈1.85–1.97, and run 3's envelope approaches
the compact limit D = 2 with the constant-velocity front characteristic of DBM.

## Provenance of each file (the "mix")

| item | source | reason |
|---|---|---|
| `data/radius_run*.csv`, `meta_*`, `radius_fits.json` | **v1** | registered frames (camera bump at t ≈ 305–320 s removed) + monotone accumulated mask (R physically non-decreasing) |
| `figures/R_vs_t_*.png` | regenerated: **v1 data, v2 presentation** | per-run figures, orange edge-censored lower-bound convention, β quoted with window systematic |
| `overlays/overlay_run1_c0.02.mp4` | **v1** | v2's run-1 overlay visibly carries the bump-artifact blobs at t ≈ 310 s |
| `overlays/overlay_run{2,3}_*.mp4` | **v2** | no bump in those runs; deposit tint, scale bar, mm readout are better sanity-check material |
| `data/fractalD_summary.csv`, `figures/fractal_run*.png` | **v2** | multi-frame × multi-threshold × two-estimator D with honest spread errors, measured at the last edge-free time |
| `data/fractal_dimensions_v1_crosscheck.json` | **v1** | independent-pipeline agreement is the main correctness argument |
| `figures/D_vs_concentration.png` | regenerated | v2's summary + v1 cross-check overplotted; axis relabelled (units are bottle labels, unverified) |
| `figures/calibration_run*.png` | **v1** | px→mm evidence |
| `data/beta_window_systematics.json` | new | β under both versions' window conventions |

Scripts live in the version folders (`version 1/scripts/week5_analysis.py`,
`version 2/scripts/*.py`); this folder holds outputs only.

## Caveats (both pipelines)

- **n = 1 per concentration** — the dominant, unquantified uncertainty in the
  D-vs-concentration trend. The week-4 videos (0.15–0.56) could extend the
  axis with the same pipelines.
- 720p limits dense-run D from above the branch scale; the unprocessed NEF
  stills (DSC_0071/0072) are the definitive check.
- Deposition current was not logged; α interpretations assume near-constant
  current.
- Concentration units as labelled on the bottles.
