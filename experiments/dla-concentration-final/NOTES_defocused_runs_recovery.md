# Fractal dimension of the "defocused" runs (0.30 / 0.45 / 0.56 %) — recovery audit

Goal: recover a reliable box-counting D for the three week-4 runs previously
excluded as "defocused blobs", or prove rigorously that it cannot be done from
the existing videos. Everything below uses the SAME estimator as the reliable
bucket (faithful mask, no hole-fill; largest/seed component; box counting with
grid-offset averaging; OLS + median local slope; window-stability sweep), with
one added honesty rule: the fit window's lower cutoff is
**max(branch width, 3·σ_blur)**, where σ_blur is measured per frame from the
deposit's own edge profile (step-edge relation g_peak = A/(σ√2π)).

Scripts: `scripts/focus_scan_defocused.py`, `scripts/fractalD_defocused.py`,
`scripts/fractalD_deblur.py`, `scripts/fractalD_perimeter.py`.
Data: `data/focus_scan.csv`, `data/fractalD_defocused.csv`,
`data/fractalD_deblur.csv`, `data/fractalD_perimeter.csv`.

## 1. What the focus scan actually found (figures/focus_scan.png)

Blur σ of the deposit edge, vs the focused 0.15 % anchor (σ ≈ 0.8 px):

| run | σ_blur over the run | verdict on optics |
|---|---|---|
| 0.56 % | **0.5–0.6 px** during growth | *sharper than the anchor* — never defocused at the deposit plane |
| 0.45 % | 1.5–1.8 px, constant | mild, ~2× anchor blur; branches visibly resolved |
| 0.30 % | 1.7 → 3.1 px, worsening as the deposit grows | genuinely blurred; the fine surface dendrites are at the blur scale |

Two further facts the scan established: the deposits keep growing well past
the nominal grounding times (the grid pitch is constant across every readable
frame → no zoom, so the late-clip enlargement is real growth), and the
deposit is static late in each clip, enabling median stacks.
The earlier "smooth blob" verdict was a frame/segmentation issue, not
irrecoverable optics.

## 2. Validations (nothing below is quoted without one)

* **Anchor through the identical pipeline.** 0.15 % grounded frame:
  D = 1.84 ± 0.07 (0.78 dec) vs the reliable-bucket 1.87 ± 0.06. Pass.
* **Synthetic blur on the anchor.** Gaussian-blurring the anchor frame to
  σ = 1.5 / 3.0 px (the 0.45 / 0.30 levels) and re-measuring with the
  max(w, 3σ) window rule leaves D unchanged (1.830 / 1.828 / 1.830):
  blur per se does not bias the slope — it *shrinks the window* (w inflates
  18 → 27 → 36 px). The 0.30 failure mode is window collapse, not slope bias.
* **Deconvolution validated at the 0.45 blur level.** Richardson–Lucy
  (Gaussian PSF, σ measured from the image's own edges, 30 iter) applied to
  the *sharp 0.56 stack blurred to σ = 1.6*: D returns 1.906 vs 1.894
  reference (Δ = +0.012). RL neither destroys nor fabricates structure at
  this blur level.
* **Deconvolution NOT validated at the 0.30 blur level.** The same test at
  σ = 2.5 collapses (the edge-σ estimator under-reads heavy blur → RL is
  under-driven → the window closes to 0.15 dec). Any RL-based 0.30 number is
  therefore quoted as indicative only.
* **Two traps found and eliminated.** (i) Single-frame RL amplifies sensor
  noise into mask speckle (local-slope std 0.27 vs 0.13) and biased the
  anchor low by 0.1 — RL is only run on median stacks. (ii) The video frames
  carry a ~10 px border vignette that RL ringing turns into a dark rim; a
  thin border sliver in the mask faked branch-width 4 px and opened a
  spurious wide window (this produced seductively stable wrong numbers,
  D = 1.86 over 1.4 "decades", in an intermediate run). Fixed by a 20 px
  crop + 4 px mask border guard; caught by the anchor's R inflating 344→803.

## 3. Results

### 0.56 % — RELIABLE, D = 1.86 ± 0.06

Grounded frame t = 198 s (σ = 0.55 px, sharper than the anchor; the lamp
switch does not matter — the reference-offset handles it).
Window [6.4 px = 0.13 mm, R/3 = 2.4 mm] = **1.26 decades**, 38 pts,
D_OLS = 1.858, D_med = 1.86, sweep stable to ±0.02 (1.84–1.89 over the
R/4–R/3 columns, k = 1–2.5). Cross-check on the 66-frame post-growth stack:
1.894. Quote **D = 1.86 ± 0.06** — same compact/dense class as 0.04–0.15 %.
(figures/fractalD_defocused_c0.56.png, fractalD_deblur_c0.56_stack.png)

### 0.45 % — RELIABLE with stated caveat, D = 1.93 ± 0.07

σ ≈ 1.6 px throughout (lamp-heated but constant). Three concordant readings:
raw grounded frame 1.91 ± 0.10 (0.67 dec); grounded-window stack + RL 1.935;
late static stack + RL **1.956 ± 0.16 over 1.14 decades, sweep ±0.03**
(RL validated at exactly this blur level, see §2). Quote
**D = 1.93 ± 0.07**, compact / effectively space-filling (D → 2 family).
(figures/fractalD_defocused_c0.45.png, fractalD_deblur_c0.45_late_rl.png)

### 0.30 % — mass-D UNRECOVERABLE as a scaling exponent

Best attainable: late static stack (t = 300–344 s, best late focus,
σ = 2.5 px) + RL → D_eff = 1.96 ± 0.18 over **0.69 decades**, and the sweep
drifts 1.93 → 2.15 as the lower cutoff rises — fails the ±0.05 stability
criterion; without RL the readings are 2.0–2.05 over ≤0.5 dec (blob limit).
Three stacked reasons make this *unrecoverable from this video*, not merely
unmeasured:

1. **Optics:** σ ≈ 2.5–3 px (≈55 µm) erases structure below ~0.2 mm, exactly
   where this run's surface dendrites live (visible early at t ≲ 60 s when
   the deposit is small, σ ≈ 1.8, but then only 0.5 dec of cluster exists).
2. **Intrinsic morphology:** the deposit is compact with coarse lobes —
   branch width ≈ 0.66 mm (31.6 px) even after deconvolution, against
   R ≈ 9.6 mm: barely one decade would exist under perfect optics.
3. **Method limit:** deconvolution is unvalidated at this blur (§2), so the
   0.69-decade RL window cannot be promoted to a reliable claim.

Honest statement for the report: *the 0.30 % deposit is compact at all
optically resolved scales (0.2–3 mm); every effective reading lies at
1.95–2.05, consistent with the space-filling D → 2 of dense growth and
clearly distinct from the open 0.02 % fractal (1.69); a finite-size effective
D cannot be resolved from 2.0 within this window.*

**Distinct fallback — boundary dimension.** The outline (1-px boundary of
the faithful mask, no RL) is resolved above 3σ and gives an *effective
perimeter dimension* D_b[3σ, R/3] = 1.26 (grounded) / 1.33 (late), with a
blur bias bounded at −0.06…−0.07 (anchor and 0.56 blurred to σ = 2.5 lose
0.069 / 0.061), so **D_b ≈ 1.3 ± 0.1**. The local slope rises with scale for
every run (the boundary is not a clean fractal either), so this is an
effective, window-stated roughness measure — a DIFFERENT quantity from the
mass D (compact mass D → 2 with a rough coast 1 < D_b < 2), reported as
such. Context: focused anchor D_b = 1.46, 0.56 % → 1.28, 0.45 % → 1.29.
(figures/fractalD_perimeter.png)

## 4. Bottom line

| conc | mass D (effective, box-counting) | window | status |
|---|---|---|---|
| 0.30 % | 1.95–2.05, unresolvable from 2.0 | <0.7 dec | **unrecoverable as exponent**; compact, D → 2; boundary D_b ≈ 1.3 ± 0.1 as distinct quantity |
| 0.45 % | **1.93 ± 0.07** | 1.1 dec (RL, validated) / 0.7 dec raw | reliable with caveat |
| 0.56 % | **1.86 ± 0.06** | 1.3 dec, no enhancement needed | reliable |

These slot naturally into the concentration series: 0.02 % → 1.69 (open DLA),
0.04–0.56 % → 1.86–1.95 (compact, finite-size readings of D → 2). The
"defocused" label survives only for 0.30 %; 0.56 % was never defocused and
0.45 %'s mild blur is fully handled by the validated window rule + RL.

Physical interpretation — why these values are sound and why they exceed the
ideal 2D-DLA 1.71 (screening-length collapse, electromigration, convection,
the 0.02 % self-control) — is logged in
`NOTES_fractal_dimension_concentration.md` §1a, next to the rest of the
report-Discussion physics.
