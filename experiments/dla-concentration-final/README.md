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
2. **`D_vs_concentration.png`** — **effective** box-counting fractal dimension vs
   concentration, **reliable (focused) runs only**: 0.02→1.69, 0.04→1.91,
   0.06→1.87, 0.15→1.87. The defocused runs (0.30/0.45/0.56) are greyed out as
   not-yet-reliable (see note below and `NOTES_fractal_dimension_concentration.md`).
3. **`growth_rate_vs_conc.png`** — mean late-time linear front speed dR/dt vs
   concentration.
4. **`R_and_dRdt_grid.png`** — one clean panel per concentration: enclosing R(t)
   [mm] with its dR/dt [µm/s] on a twin axis, plus an all-runs R(t) overlay in
   the 8th slot. Polished replacement for the 7-panel working draft.
5. **`D_vs_concentration_with_crops.png`** — the reliable D-vs-concentration plot
   with a **4-over-3 gallery of the grounded video frames** below it (frame at
   each run's `t_measured_s`, cropped to a square of side 2·1.15·R, 1 mm scale
   bar). The four **focused** crops are captioned with their reliable D and show
   sharp branch structure; the three **defocused** crops are greyed and marked
   "defocused" — visibly blurred blobs, which is *why* their D is not reliable.
   The sparse open cluster (0.02 %, D≈1.7) vs the dense morphologies (≥0.04 %,
   D≈1.9) is directly visible. Crops in `figures/crops/`.
6. **`screening_masks.png` / `screening_crops.png`** — *open-vs-dense comparison*
   (Discussion figure for why D > DLA 1.71): two clean panels, our real 0.02 %
   deposit (open, D=1.69 ≈ DLA) vs 0.06 % (dense, D=1.87 → compact), with mm axes
   and a minimal conc+D tag. Two versions — **masks** (black-on-white segmented
   structure) and **crops** (colour video frame). All explanatory text lives in
   the report caption, not on the figure. Built by `screening_figure.py`.

## Method decisions (read before quoting numbers)

**Fractal D — the reliable bucket (focused runs only).** After an end-to-end
audit (see `NOTES_fractal_dimension_concentration.md`), D is measured by
**box-counting on a faithful mask** (interior hole-fill removed), fit over the
window **[branch width, R/3]** (above the branch width, below finite-size),
**verified by a window-stability sweep** (D stable to ±0.05). This is only
trustworthy for the four **focused** runs; the values are in
`data/fractalD_reliable.csv`:

| conc | **D (reliable)** | old pipeline | focus |
|---|---|---|---|
| 0.02 | **1.69 ± 0.05** | 1.611 | focused |
| 0.04 | **1.91 ± 0.05** | 1.887 | focused |
| 0.06 | **1.87 ± 0.04** | 1.887 | focused |
| 0.15 | **1.87 ± 0.06** | 1.886 | focused |
| 0.30 / 0.45 / 0.56 | **— (unreliable)** | 1.89 / 1.91 / 1.87 | **defocused** |

Report D as an **effective** dimension over its stated window. **0.02 % is a
genuine open DLA-like fractal (~1.7)**; 0.04–0.15 % are **compact / effectively
space-filling (D → 2, read as ~1.9 at finite size)** — *not* genuine 1.9
fractals. The plateau is the D = 2 ceiling plus limited resolving power; the
driver is **concentration** (not voltage). Defocused runs are excluded (mask =
smooth blob); box-counting (no centre) is preferred over sandbox for these
anisotropic clusters. Corrected values match the original pipeline to ±0.05, so
the earlier interior-hole-fill artefact was real but small.

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
