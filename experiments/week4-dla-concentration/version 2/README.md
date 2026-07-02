# Week 4 — DLA vs CuSO₄ concentration (high-concentration series)

Cu electrodeposition in the quasi-2D cell at **12 V**, varying only the CuSO₄
concentration: **0.56 / 0.45 / 0.30 / 0.15** (runs 1–4, all 2026-06-18).
Same pipeline and deliverables as
[`week5-dla-concentration/version 2`](../../week5-dla-concentration/version%202/README.md)
(see there for the base method and its validation); this README documents the
week-4 specifics. Raw videos live in `experiments/week4-dla-no-shlomo/`
(outside git). Calibration from the in-frame mm grid: **47.8–49.5 px/mm**,
per run.

## Run conditions (experimenter's log — matters for interpretation)

A lamp was added for better lighting (lower ISO), which **also heated the
cell** — an unintended temperature component. Lamp-heated deposit turns
red/copper-coloured; unheated stays black.

| run | conc. | lamp | focus | notes |
|---|---|---|---|---|
| 1 | 0.56 | OFF until t≈360 s, then ON (in-clip brightness jump) | defocused early | M(t) unreliable after 360 s (lamp shifts the change-reference); all fits close before |
| 2 | 0.45 | ON from start | mostly defocused | heated throughout — not strictly comparable to lamp-less runs |
| 3 | 0.30 | none | defocused | kinetic exponents blur-biased |
| 4 | 0.15 | none | **focused** | **anchor run of the series** |

`run4_0.15.mov` is byte-identical to `DSC_0036.mov`. The unnamed DSC clips are
continuations/tests (camera-clock mapping): run 1 ← DSC_0025, run 2 ← DSC_0030,
run 3 ← DSC_0033/0034, run 4 ← DSC_0037 + DSC_0040 (gaps of only 16 s / 43 s);
DSC_0019–0022 / 0026 / 0027 / 0031 are setup or refocus tests. Continuation
clips start with the deposit already present, so the temporal-change
segmentation does not transfer to them as-is; they are not analysed here.

## Results

| | run 1 — 0.56 | run 2 — 0.45 | run 3 — 0.30 | run 4 — 0.15 |
|---|---|---|---|---|
| nucleation t₀ | 7 s | 4 s | 6 s | 3 s |
| early radius law β | 0.35 | 0.32 | 0.31 | 0.37 |
| late front (linear, pre-edge fit, R²>0.975) | 28.2 µm/s | 30.3 µm/s | 23.1 µm/s | 22.3 µm/s |
| kinetic M∝Rg^D | 1.91 | 2.09 | (2.33 — blur-biased) | 2.07 |
| **fractal D (stills)** | **1.88 ± 0.04** | **1.97 ± 0.08** | **1.89 ± 0.03** | **1.88 ± 0.03** |
| first edge contact | 198 s | 149 s | 139 s | 212 s |
| final enclosing R (lower bound) | 11.9 mm | 10.2 mm | 11.1 mm | 7.5 mm |

**The finding:** at 0.15–0.56 the measured dimension is **flat at D ≈ 1.9** —
the morphology has saturated in the dense/compact regime, and every run shows
the dense-branching signature of a **constant-velocity envelope** after an
early transient (front speeds fit strictly before first frame-edge contact) (the β values fit the transient; the honest late-time model is
linear R(t)). Nucleation is near-instant. Combined with week 5
(0.02 → 0.06: D = 1.54 → 1.97), the full series reads: D rises steeply from
1.54 at 0.02, crosses the 2D-DLA 1.71 around ~0.03–0.04, and plateaus at ≈1.9
from 0.06 through 0.56.

**How to quote D (window-scan verdict, `figures/windowscan_*.png` +
`scripts/fractalD_window_scan.py`).** The measured D is *window-stable* in
every run — refitting over a grid of physical windows (s_min 0.08–0.4 mm ×
s_max 0.6–2.5 mm) moves it only within 1.84–1.94, with no drift toward 1.71;
a 1-px mask erosion shifts it ≤ 0.05; and the mass-radius occlusion correction
on/off is indistinguishable. That rules out fit-window choice, edge dilation
and correction residuals as sources of the offset above 1.71. But the sharper
diagnostic — the local slope d(logN)/d(logs) — shows a true scale-free plateau
only for week 5's sparse 0.02% cluster (D ≈ 1.6, the control proving the
pipeline reads sub-1.71 when the morphology is sparse). For 0.04–0.56 the
local slope declines with scale (≈1.95 at 0.1 mm → ≈1.8 at 0.5–0.6 mm) without
settling before finite-size noise: the quoted values are an **effective
dimension at the accessible 0.1–1 mm scales of a morphology still crossing
over from compact toward fractal** — consistent with migration/convection-
driven dense-branching growth (independently evidenced by the linear fronts),
with the asymptotic regime unreached at these cluster sizes. Branch widths
(distance-transform ridge): 0.16 / 0.61 / 1.32 / 0.70 mm for runs 1–4.

**Kinetic vs static D — a real distinction, not a bug.** Run 4 (clean) gives
kinetic M∝Rg^2.07 along the growth trajectory but static D = 1.88 from the
frame geometry: at these concentrations the interior keeps densifying while
the envelope advances, so growth is **not self-similar** and the trajectory
exponent exceeds the geometric dimension. (Week 5's sparse clusters grew
self-similarly and the two agreed.) The static box-counting/mass-radius values
are the quoted D; run 3's kinetic 2.33 is this effect compounded by defocus.

**Open question vs week 3:** week 3 measured D = 1.65 ± 0.04 at 0.29% — well
below this plateau at the same nominal concentration and voltage, but in a
different cell/session with a much sparser deposit. Not papered over here;
worth a discussion paragraph in the report (cell geometry / layer thickness /
effective current density differ).

## Week-4 pipeline additions (each set by a measured gap, on top of week 5)

1. **Absolute-darkening pathway** — a defocused deposit is a smooth dark blob:
   blur destroys local contrast (the flat-field hysteresis saw only the sharp
   fringe, which mis-seeded and mis-centred run 1's early circle) but not
   absolute darkening. Pixels darkened > 40 gray levels vs the reference enter
   the mask directly (measured: deposit 55–92, shadow/halo 15–40).
2. **Interior hole-filling** — the filled deposits are far wider than the
   flat-field scale, so interiors have no local contrast. Enclosed non-mask
   regions whose median darkening ≥ 15 are filled where darkened > 40;
   "outside" requires genuinely undarkened paper, so the boundary drop-shadow
   (which darkens as much as interiors) stays out topologically.
3. **See-through tint rule (wire v > 185)** — the glow *tints* the deposit
   green (s≈124, v≤173) but the deposit is plainly visible through it; only
   the opaque bright wire/glow (v≥198 over paper) is excludable. Measured gap
   173/198.
4. **Glare occluder** — the lamp-lit glow core (saturated s>110, bright
   v>140, and *smooth*, local-std < 4.5 — brightly-lit deposit passes the HSV
   cut but is textured at 10–13) joins the wire in the mass-radius occlusion
   correction and conducts cluster connectivity. In lamp-less run 4 the dim
   glow wedge (s=104, v=139) misses these cuts and its annuli go uncorrected —
   a measured +0.02 shift on D, inside the quoted ±0.03.
5. **Multi-frame calibration** — the grid strip can be unreadable in a single
   frame; several sample times are tried (and the base-peak window widened to
   20–70 px for this week's zoom).
6. **Continuity of the enclosing circle** (v2.1, after review caught steps in
   R(t)): (a) a pixel darkened > 40 gray levels cannot be opaque glare — the
   heated copper deposit shares the glare's colour+smoothness signature and a
   whole crescent was being hidden until hole-filling enclosed it, stepping R
   at run-2 t = 179.5 s; (b) a strong-cored component inside the aggregate
   disc is admitted the moment it appears (connectivity through occluded
   corridors can lag); (c) the circle and the edge flag are computed over the
   cluster MEMORY — the deposit is permanent, so R is non-decreasing by
   construction, and marginally-darkened frontier regions can no longer
   oscillate it. Result: zero R decreases in all four runs; the single
   remaining step (+0.8 mm, run 2 t = 187.5 s, inside the edge-censored
   lower-bound regime) is a first-detection event of a semi-transparent front
   measured at darkening 31 (< the 40 threshold) two frames earlier — steps in
   the faded regime are occlusion/detection reveals, not growth events.
   (A 40/25 darkening hysteresis was tried and rejected: it floods the
   drop-shadow ring, inflating the focused run's mask by 42%.)

Verification battery as in week 5, per run: R(t)/M(t) monotone (worst R dip
0.3 mm, on the defocused run 2), zero mass only before t₀, overlay stills
inspected at nucleation / mid / late / the run-1 lamp switch. Run 1's mass
dips ~25% at t = 368–384 s when the lamp brightens the scene — quantified,
outside all fit windows, R(t) unaffected.

## Layout & how to run

Same as week 5: `scripts/` (`enclosing_radius.py`, `fractal_dimension.py`,
`summary_from_csv.py`), `data/` (per-frame CSVs + `fractalD_summary.csv`),
`figures/` (`R_vs_t_*`, `fractal_*`, `D_vs_concentration.png`, `R_vs_t_all.png`),
`overlays/` (×15-speed circle-overlay mp4s; deliverable 2).

```
python scripts/enclosing_radius.py      # R(t) + CSVs + overlays (~40 min)
python scripts/fractal_dimension.py     # D per concentration
python scripts/summary_from_csv.py      # combined figure from CSVs
```

`WEEK4_VIDEO_DIR` overrides the raw-video location. Figure captions as in the
week-5 README.
