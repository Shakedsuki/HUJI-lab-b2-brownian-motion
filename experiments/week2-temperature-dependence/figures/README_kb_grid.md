# Week-2 Boltzmann constant — per-run k_B grid

**Deliverable:** `kb_grid.png` — Stokes–Einstein D-vs-(1/r) panels, one run per
*measured starting temperature*. The per-panel headline k_B is the robust
**per-bead median** of 6πη(T)·r·D/T — NOT the through-origin slope: with the
narrow ~0.8–1.4 µm⁻¹ lever arm the slope is estimator-unstable (run7 is 1.00× by
slope, 1.26× by median) and the median never regresses on 1/r, so radius x-scatter
cannot bias it. The slope is kept only as a faint cross-check. The **synthesis
figure is `kb_summary.png`** (k_B/k_B^acc vs T + the near-ambient model).

## Headline result (FINAL — all 16 runs processed; 10 in grid)

**k_B = (1.53 ± 0.07_stat) × 10⁻²³ J/K = 1.10× accepted**, n = 87 clean free
beads, 10 runs, 6 temperatures. Common-mode systematics (do **not** average down):
**radius offset ±1 px = ±0.16×**, T-label ±1 °C ≈ ±0.03×. Sensitivity including
drift beads → 1.15×.

| run | T (°C) | n | k_B median | χ²/dof | slope x-check |
|-----|--------|---|-----------|--------|---------------|
| run2 | 14.0 | 5 | 1.39× | 38 | 1.15× |
| run3 | 14.0 | 4 | 1.01× | 3.5 | 1.09× |
| run4 | 14.0 | 8 | 1.27× | 1.6 | 1.23× |
| run5 | 15.2 | 13 | 1.34× | 31 | 1.26× |
| run7 | 16.8 | 16 | 1.26× | 9.7 | 1.00× |
| run8 | 16.8 | 9 | 1.15× | 8.9 | 1.17× |
| run9 | 20.0 | 5 | 0.92× | 1.5 | 1.00× |
| run13 | 24.3 | 5 | 1.52× | 1.2 | 1.52× |
| run15 | 30.3 | 7 | 0.91× | 63 | 0.80× |
| run16 | 30.3 | 15 | 0.74× | 53 | 0.70× |

**χ²/dof ≫ 1 on most runs is itself a result:** the per-bead bars (σ_D ⊕ σ_r=1 px)
explain only a fraction of the bead-to-bead spread — forcing χ²/dof→1 needs
σ_r ≈ 6.5 px (absurd for tagging), so the spread is real structure, not Gaussian
noise. The honest per-run/pooled uncertainty is the scatter, not the bars.

**Two systematics (see kb_summary.png):** (1) a temperature-label trend — extracted
k_B falls with nominal T because the fluid sat near ambient regardless of setpoint
(fitted decoupling f = 0.72 ± 0.19 → warm "30.3 °C" sample ≈ 23 °C); (2) a
common-mode radius offset (±1 px → ±15%) that can shift the whole dataset. Caveat:
**24.3 °C now rests on run13 alone (n=5, high at 1.52×)** — run14 dropped as
non-stationary; that point is under-constrained.

**Excluded** (all in `kb_grid.EXCLUDED_RUNS`, reasons printed at every rebuild):
run6 (convection, 5/8 drift), run10+run11 (residual drift, ~165 nm/s), run12
(convection), run14 (non-stationary — full-clip D(t) wanders, 64 px drift, no
plateau); run1 below the 3-clean-bead gate.

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
