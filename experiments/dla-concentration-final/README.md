# DLA vs CuSO₄ concentration — combined report figures (final)

Report-ready figures for the DLA-concentration extension, **merging both
measurement sessions into one concentration series** (0.02 → 0.56 %):

| session | runs | regime |
|---|---|---|
| [week 5 v2](../week5-dla-concentration/version%202/README.md) | 0.02 / 0.04 / 0.06 % | sparse, focused anchor |
| [week 4 v2](../week4-dla-concentration/version%202/README.md) | 0.15 / 0.30 / 0.45 / 0.56 % | dense / compact plateau |

All figures are rebuilt **from the committed per-run `radius_*.csv` and the two
`fractalD_summary.csv` files** — no video decode. `scripts/report_figures.py`
runs in seconds and is the single source for everything in `figures/`. It reads
the upstream weeks' `data/` directories directly; nothing is duplicated here.

```
python scripts/report_figures.py      # writes all four figures to figures/
```

## Figures (report deliverables)

1. **`fill_fraction_vs_conc.png`** — *sanity check.* Deposit occupancy of the
   enclosing disc, φ = M / (πR²), vs concentration. Point = median over the
   developed (larger-half) edge-free frames; band = 16–84 percentile.
2. **`D_vs_concentration.png`** — fractal dimension vs concentration, **both
   sessions, box-counting only** (see note below).
3. **`growth_rate_vs_conc.png`** — mean late-time linear front speed dR/dt vs
   concentration.
4. **`R_and_dRdt_grid.png`** — one clean panel per concentration: enclosing R(t)
   [mm] with its dR/dt [µm/s] on a twin axis, plus an all-runs R(t) overlay in
   the 8th slot. Polished replacement for the 7-panel working draft.
5. **`D_vs_concentration_with_crops.png`** — the D-vs-concentration plot with a
   **4-over-3 gallery of the grounded video frames** below it: for each run, the
   frame at its `t_measured_s` (the fully-developed frame the box-counting D was
   grounded on), cropped to a square of side 2·1.15·R centred on the enclosing
   circle, with a 1 mm scale bar and a border matching the point's colour. The
   sparse D≈1.6 cluster (0.02 %) vs the dense D≈1.9 morphologies (≥0.04 %) is
   directly visible. Crops saved in `figures/crops/`.

## Method decisions (read before quoting numbers)

**Fractal D is box-counting only.** The upstream pipeline quoted D as the mean
of two estimators (box-counting + mass-radius); per the report split we drop
mass-radius and quote **`D_boxcount`** straight from each week's
`fractalD_summary.csv`. Effect on the quoted values:

| conc | old (box+mr mean) | **box-only** | Δ |
|---|---|---|---|
| 0.02 | 1.517 | **1.611** | +0.094 |
| 0.04 | 1.822 | **1.887** | +0.065 |
| 0.06 | 1.951 | **1.887** | −0.064 |
| 0.15 | 1.880 | **1.886** | +0.006 |
| 0.30 | 1.888 | **1.892** | +0.004 |
| 0.45 | 1.974 | **1.906** | −0.068 |
| 0.56 | 1.879 | **1.866** | −0.013 |

Dropping mass-radius mainly tightens the two runs where the estimators disagreed
most (0.02 sparse; 0.45 defocused+lamp-heated). The qualitative story is
unchanged: **D rises from ~1.6 at 0.02 %, crosses the 2-D DLA value 1.71 by
~0.04 %, and plateaus at ≈1.89 through 0.56 %.**

**Error bar on D = ±0.03 (systematic floor).** The per-frame box-counting
scatter is not recoverable from the summary CSVs (it needs the video frames), so
the plotted bar is the pipeline's documented systematic floor. Independent
window-scan stability is ±0.05 (week-4 README). If a purely statistical
box-only error bar is wanted, re-run the upstream `fractal_dimension.py` against
the videos and record `std(Ds_bc)`.

**Growth rate** = slope of a straight-line fit to R(t) [mm] over the edge-free
frames past the early transient (upper 0.35–0.98 of each run's size), ×1000 →
µm/s. Agrees with the week-4 README's hand-quoted late fronts (e.g. 0.56 %:
28.0 vs 28.2 µm/s; 0.15 %: 22.4 vs 22.3). The plotted bars are the formal fit
±1σ and **understate** the true uncertainty (the reported R is smooth/cap-
limited, so the fit is artificially tight) — treat them as a lower bound.

## Caveats worth a line in the report

- **φ is size-dependent.** For a fractal M ∼ R^D so φ = M/R² ∼ R^(D−2) is not a
  constant; this figure compares occupancy *at each run's own accessible cluster
  size*, not at a fixed size. The sparse 0.02 % run (φ ≈ 0.16) sits far below the
  dense plateau (φ ≈ 0.6), consistent with its low D — but the near-flat plateau
  from 0.06–0.45 and the slight dip at 0.56 % mean φ is **not strictly monotone**
  in concentration; it saturates, mirroring the D plateau. Don't over-claim a
  linear φ(conc).
- **The dR/dt plateau at ≈90 µm/s early in the dense runs is the continuity cap**
  (3× the fastest front) in the reported envelope, not a physical constant —
  growth-law exponents upstream use the uncapped `circ_R_raw_px`, so the cap
  never enters a fit.
- **0.56 % is the weakest run** (lamp off→on at t≈360 s, defocused early); its
  low-side points in both the D and growth-rate figures are consistent with
  that, not necessarily a real down-turn at high concentration.
- **The snapshot crops show the raw frame, not the segmentation.** The green
  glow and grey electrode/wire visible in each crop are *excluded* from the
  box-count by the occluder mask — D is grounded on the black deposit only. The
  crops are for morphology context; they are not the binary the fit ran on.

## Video dependency (only figure 5)

Figures 1–4 are CSV-only and regenerate anywhere. **Figure 5's crops read pixels
from the raw videos, which are outside git** (`week4-dla-no-shlomo/`,
`week5-dla-concentration/raw-videos/` — override with `WEEK4_VIDEO_DIR` /
`WEEK5_VIDEO_DIR`). The grabbed crops in `figures/crops/` and the composite PNG
are committed as artifacts, so the figure survives without the videos; only
re-grabbing needs them (and `ffmpeg`). Frame times and clip names are pinned in
the `RUNS` table in `scripts/report_figures.py`.
