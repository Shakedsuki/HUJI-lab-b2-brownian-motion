# Week 5 — DLA vs CuSO₄ concentration (with Shlomo)

Cu electrodeposition in the quasi-2D cell at **12 V** (same as week 3), varying
**only the CuSO₄ concentration**: run 1 = **0.02**, run 2 = **0.04**,
run 3 = **0.06**. Each run was filmed close-up (1280×720 @ 59.94 fps,
11–16 min); the raw videos live in
`experiments/week-5-last-measurements-with-shlomo/` (outside git).
Millimetre graph paper in frame gives a real px→mm calibration:
**35.5–36.0 px/mm**, measured per run from the autocorrelation harmonics of
the blue grid-line profile (±0.5%; camera fixed between runs).

Instructor deliverables:

1. **enclosing-circle radius vs time** per sample → `figures/R_vs_t_*.png`,
   raw per-frame series in `data/radius_*.csv`;
2. **overlay video** of that circle on the footage → `overlays/overlay_*.mp4`
   (×15 speed: sampled at 2 Hz, played at 30 fps);
3. **fractal dimension per concentration** → `figures/D_vs_concentration.png`,
   `data/fractalD_summary.csv`.

## Results

| | run 1 — 0.02 | run 2 — 0.04 | run 3 — 0.06 |
|---|---|---|---|
| morphology | sparse 3-arm dendrite | dense radial | near-compact disc |
| nucleation t₀ | 13 s | 12 s | 31 s |
| radius law (early) | R ∝ t^**0.51±0.03** | R ∝ t^0.37±0.01 | power law fails; late front **linear, ≈26 µm/s** |
| mass law α (M ∝ t^α) | 0.87 | 0.75 | 0.53 |
| kinetic D (M ∝ Rg^D) | 1.57 ± 0.02 | 1.83 ± 0.02 | 1.99 ± 0.05 |
| **fractal D (stills)** | **1.54 ± 0.09** | **1.89 ± 0.05** | **1.97 ± 0.21** |
| first frame-edge contact | 400 s | 600 s | 418 s |
| final enclosing R (lower bound) | 9.8 mm | 13.7 mm | 15.4 mm |

**The finding:** concentration changes the *morphology class*, not just the
rate. At 0.02 the growth is DLA-like — anisotropic branches, R ∝ t^½ (DLA
predicts t^(1/D)), D well below the 2D-DLA 1.71. With increasing concentration
D rises through ≈1.9 and saturates at ≈2, and the envelope crosses over to the
**constant-velocity compact front** of the dense-branching regime — run 3's
R(t) is linear over its last 500 s. Week 3 (0.29%, same voltage) gave
D = 1.65 ± 0.04, between the trend's low end and its plateau.

The quoted fractal D per concentration is the mean over
{3 late frames} × {3 segmentation thresholds} × {box-counting, mass-radius}
(18 estimates); its error is the measured spread ⊕ a ±0.03 threshold
systematic — **not** the (much smaller) fit error. The independent kinetic
exponent D(M~Rg), measured from the whole growth history rather than a single
frame, agrees with the stills for all three runs — the main correctness check.
Run 3's ±0.21 is honest: its mass-radius estimate (2.11) exceeds 2 because the
compact deposit's centre is occluded (wire + bubble); box counting (1.84) and
the kinetic route (1.99) bracket the compact limit.

## Method (scripts/enclosing_radius.py)

Per sampled frame (2 Hz), the *growing* deposit is isolated as in week 3 —
temporal change vs a median-of-first-frames reference + flat-field
local-contrast hysteresis + HSV wire mask — with four week-5 additions, each
introduced after a verification check failed and traced to a physical cause:

1. **Blue-grid exclusion** — the mm-paper lines (B ≫ G,R) can never count as
   deposit, whatever the lighting does.
2. **Green-only wire hue (H 25–95)** — week 3's band (18–95) also caught the
   copper-brown central deposit and punched a hole in the mask.
3. **Strong-core gate** — a component must contain a truly dark pixel
   (local-contrast score ≥ 0.25). The wire's *moving shadow* darkens paper
   specks enough to pass the change+hysteresis test but only weakly
   (measured: specks ≤ 0.21, real deposit ≥ 0.38); without this gate they
   chain into the cluster and balloon the enclosing circle.
4. **Cluster memory + wire connector** — the deposit is immobile and
   permanent, so (a) the dilated wire mask conducts *connectivity* (not mass)
   between the pieces it visually splits, but only within 24 px of existing
   deposit, and (b) any component that has ever belonged to the cluster stays
   in it. This keeps branches whose root crosses the static wire-shadow band
   (invisible to the change mask) from flickering out.

The **enclosing circle** is `cv2.minEnclosingCircle` of the seed-connected
cluster (convex hull first — exact and fast). Frames where the deposit touches
the frame border are flagged `edge`: from first contact the circle is a lower
bound (orange in figures and overlays; excluded from every fit). Growth-law
exponents are fitted on the objective window of week 3 (mass 8–75% of plateau,
t from observed nucleation, Hampel outlier rejection), with a fit-window-scan
systematic.

**Verification battery** (all three runs pass): R(t) and M(t) non-decreasing
(worst residual blip 0.6 mm — tip flicker in a max-statistic); zero mass only
before t₀; overlay circle visually tight at nucleation / mid / late / all
former-artifact times; two independent code paths agree on the same frame's
cluster; α/β ≈ D self-consistency to ~10% (gap = the diffuse-halo mass bias,
quantified by the threshold scan).

## Fractal dimension (scripts/fractal_dimension.py)

Run AFTER the kinetics pass (reads its CSVs for the last edge-free time and
seed). Per run: 3 late frames (100/85/70% of the last edge-free time) × 3
hysteresis thresholds (0.12/0.15/0.18), each giving

* **box counting** N(s) ∝ s⁻ᴰ, fitted for 8 px ≤ s ≤ R/8;
* **mass-radius** M(<r) ∝ rᴰ about the seed, 30 px ≤ r ≤ 0.8 R, each annulus
  corrected for wire occlusion (annuli >60% occluded dropped).

Components are disc-gated (inside 1.15 R₉₉ of the seed) rather than
seed-connected — single frames carry no cluster memory.

## Run

```
python "version 2/scripts/enclosing_radius.py"        # R(t) + CSVs + overlays (~40 min)
python "version 2/scripts/fractal_dimension.py"       # D per concentration  (~5 min)
python "version 2/scripts/summary_from_csv.py"        # combined figure from CSVs only
```

`WEEK5_VIDEO_DIR` overrides the raw-video location.

## Figure captions

**`R_vs_t_<run>.png`** — Radius of the minimal circle enclosing the deposit vs
time, in mm (grid-paper calibration). Blue: trustworthy measurements; orange:
the deposit touches the frame border, so R is a lower bound. Black: power-law
fit R ∝ (t−t₀)^β over the pre-saturation growth window; the quoted error
combines the statistical slope error with the fit-window systematic.

**`overlay_<run>.mp4`** — The deliverable-2 sanity check: enclosing circle
(red; orange once edge-censored) drawn on the footage at ×15 speed, with the
segmented deposit tinted green, the growth seed (magenta cross), circle centre
(dot), R readout in mm and a 5 mm scale bar.

**`fractal_<run>.png`** — Left: segmentation of the measured late frame (red =
deposit, blue = excluded wire, dashed = R₉₉). Centre: box counting, slope −D
over the scale-free window (red points). Right: occlusion-corrected
mass-radius about the seed, slope +D (blue points). Both fit windows exclude
the branch-width and finite-size regimes.

**`D_vs_concentration.png`** — The deliverable-3 summary: combined D (±
spread ⊕ systematic) vs concentration, with the individual estimators and the
2D-DLA reference 1.71. The rise 1.54 → 1.89 → 1.97 and the run-3 saturation at
the compact limit D = 2 is the concentration → morphology-transition result.
