# Week-2 Boltzmann constant — per-run k_B grid

**Deliverable:** `kb_grid.png` — a grid of Stokes–Einstein D-vs-(1/r) panels, one
run per *measured starting temperature* (filename temp, ±1 °C nominal). The slope
of each panel is k_B:

    D = (k_B T / 6π η(T)) · (1/r)      ⇒      k_B = slope · 6π η(T) / T

## Headline result (FINAL — all 16 runs processed; 12 in grid, n=102 clean beads)

**k_B = (1.40 ± 0.07) × 10⁻²³ J/K = 1.01× accepted**
(pooled per-bead median; free diffusers, significant-drift beads excluded;
ALL clean tagged runs pooled — same-T duplicates are reproducibility evidence)

| run | T (°C) | n clean | k_B (fit) | R² |
|-----|--------|---------|-----------|-----|
| run2 | 14.0 | 5 | 1.15× | 0.87 |
| run3 | 14.0 | 4 | 1.09× | 0.94 |
| run4 | 14.0 | 8 | 1.23× | 0.98 |
| run5 | 15.2 | 13 | 1.26× | 0.86 |
| run6 | 15.2 | 3 | 1.07× | 0.95 |
| run7 | 16.8 | 16 | 1.00× | 0.81 |
| run8 | 16.8 | 9 | 1.17× | 0.93 |
| run9 | 20.0 | 5 | 1.00× | 0.98 |
| run13 | 24.3 | 8 | 1.08× | 0.75 |
| run14 | 24.3 | 9 | 0.69× | 0.83 |
| run15 | 30.3 | 7 | 0.80× | 0.87 |
| run16 | 30.3 | 15 | 0.70× | 0.82 |

Sensitivity: including the 28 drift-flagged beads → 1.10×.
Same-T reproducibility: 14 °C 1.09–1.23×; 15.2 °C 1.07–1.26×; 16.8 °C
1.00–1.17×; 24.3 °C 0.69–1.08× (the widest split); 30.3 °C 0.70–0.80×.
Run-to-run spread ±0.2–0.3× is the dominant systematic.
Excluded: run1 (only 2 clean beads); run10, run11, run12 discarded on drift /
convection evidence (see EXCLUDED_RUNS in kb_grid.py — reasons print at every
rebuild).

## Measurement policies (in the order applied)
1. **Hand-tagged radii** (`radius_manual.csv`, via `radius_tag.py`): the auto
   outer-edge radius over-reads by a measured ~0.30 µm (diffraction), which would
   push nearly every bead past r* — so only hand-tagged runs enter the grid.
2. **Wall-pinned removed:** beads with r > r*(T) (sedimentation cut) are dropped.
3. **Significant-drift excluded (headline policy):** beads with residual drift
   |v| > 0.1 µm/s AND |v| > 2σ_v (σ_v = √(2D/T_span), the finite-track noise
   floor) are excluded from every fit/pool; shown as open red markers.
4. **min 3 clean beads** per run (`--min-free`), else the run is excluded.
5. **Documented run discards are QUALITY-only** (`kb_grid.EXCLUDED_RUNS`, reason
   printed at every rebuild): run12 (convection: QA flow 2× all other runs, 7/12
   beads drift-flagged, 1.7× vs clean same-T run13's 1.08×). Same-T duplicates
   are POOLED, not dropped.

## Known systematic (do NOT retro-tune)
The warm end reads low (run16+run15 agree: beads at "30.3 °C" diffuse like ~22 °C
water), the cold end slightly high — consistent with the sample sitting closer to
ambient than the stage thermometer label during far-from-ambient setpoints.
Temperatures are NOT adjusted; the T-label is carried as the dominant systematic.
Runs recorded nearest ambient (run7, run9, run13) read 1.00–1.08×.

## Per-bead errors (each panel)
σ_D from the MSD-fit covariance (y bars), σ_(1/r) = σ_r/r² with σ_r ≈ 1 px (x
bars), through-origin fit R² (uncentered), k_B ± max(fit-covariance, per-bead
scatter) — both in `kb_grid_summary.csv` — plus robust median cross-check.

## Add / re-tag a run
1. `python process_all.py runX`   # track + curate (already done for most)
2. `python radius_tag.py runX`    # GUI; preloads previous tags for refinement
3. `python kb_grid.py`            # rebuild grid + sweep (+ `finalize.py` for per-bead figs)

## Other outputs
- `kb_sweep_uniform.png` — k_B vs T (invariance) + D·r vs T (SE temperature law).
- `kb_grid_summary.csv` — per-run table (both error estimates, R², drift counts).
- `measurements/<run>/pipeline/plot1_perbead.png` — per-bead MSD fits + R² +
  ⟨Δx⟩/⟨Δy⟩ drift insets; `msd.csv` / `drift_perbead.csv` carry σ_D, R², v ± σ_v, flags.
