# Critical assessment of the final-report figures (vs literature & predictions)

Skeptical pass over the four main figures (2026-07-13, after the 0.30 %
exclusion and the 0.45/0.56 % recovery). What holds up, what is artifact,
what must not be over-claimed.

## 1. D_vs_concentration_with_crops — holds up, with two reading rules

**Matches literature.**
- 0.02 % → 1.69 ± 0.05 agrees with classic electrodeposition (Matsushita 1984:
  1.66 ± 0.03) and 2D DLA (Witten–Sander 1.71). This point doubles as the
  series' control: the pipeline reproduces the DLA limit where the physics is
  diffusion-limited.
- The jump to a flat 1.86–1.93 band by 0.04 % matches the morphology diagram's
  concentration axis (Sawada 1986; Grier 1986): dilute → open fractal,
  concentrated → dense branching, whose true dimension is D → 2 (Ben-Jacob
  1986); ~1.9 is the expected finite-size reading.
- The crops corroborate qualitatively: bare paper between branches at 0.02 %,
  narrow fjords everywhere else.

**Reading rules (do not over-claim).**
- Our concentration sampling cannot resolve the open→dense transition; claim
  only "the transition lies below 0.04 % at our drive", not its shape.
- Do NOT rank values inside the dense band (1.86 vs 1.93): window systematics
  overlap, and with ~1 decade of scaling 1.9 is indistinguishable from 2.0
  (Malcai/Avnir 1997). The 0.45 % error bar touching 2.0 is fine — the claim
  IS "D → 2".
- The visual openness of the 0.04 % crop vs the denser-looking 0.15/0.45 %
  at the same D ≈ 1.9 is expected: near the ceiling, mass-D saturates while
  branch texture still varies — D is a blunt instrument in the dense band.
- Crops show the raw frame (glare, wire); the fit ran on the occluder-masked
  binary, not on what the crop shows.

## 2. fill_fraction_vs_conc — consistency panel ONLY

- φ is NOT independent of D (φ = M/πR² and M ∝ R^D ⇒ φ ∝ R^(D−2);
  measured corr(φ, D) = 0.994). Present as a consistency check, never as
  corroboration.
- The step 0.16 → ~0.6 and plateau mirror the D step and ceiling —
  consistent, as it must be.
- 0.02 % quantitative check: naive (w/R)^(2−D) with D = 1.69 gives ~0.3;
  measured 0.16 is lower, in the direction expected from the enclosing
  CIRCLE overestimating the area of an anisotropic cluster. Direction fine,
  don't quote quantitative agreement.
- The dip at 0.56 % (0.56 vs 0.64 at 0.45 %) is NOT established physics: it
  is within run-to-run envelope-geometry systematics (a lopsided deposit
  deflates φ; 0.56 % also has the finest branches, w ≈ 0.13 mm). φ is not
  strictly monotone — do not claim a trend reversal.

## 3. growth_rate_vs_conc — the most physics-rich panel, one confound

- **The strongest quantitative observation:** concentration spans 28×
  (0.02 → 0.56 %) but the front speed rises only ~2.3× (12–14 → 28–32 µm/s).
  This grossly sub-linear response rejects a naive "rate ∝ ion supply"
  picture and is consistent with the migration-controlled thin-cell result
  that the dense front advances at roughly the anion drift velocity
  v ≈ μ_a·E (Chazalviel space-charge model, PRA 42, 7355 (1990); Fleury,
  Rosso, Chazalviel experiments, early 1990s) — the speed is set by field and
  mobility, only weakly by concentration. **Verify these citations against
  the originals before formal use.**
- **Confound to disclose:** the 0.45 % run was lamp-heated throughout;
  thermal convection enhances transport, so part of its 32 µm/s peak may be
  heating, not concentration. The 0.45 → 0.56 "down-turn" (32 → 28) is
  therefore not evidence of a real maximum.
- The 0.02/0.04 inversion (14.2 vs 12.1) is within run-to-run systematics
  (cell gap, wire geometry) — the plotted ±1σ fit bars badly understate true
  uncertainty (reported R is smoothness-capped; treat bars as lower bounds).
- Net defensible claim: growth rate increases weakly and roughly
  monotonically with concentration, consistent with migration-limited
  growth; individual point orderings are not significant.

## 4. R_and_dRdt_grid — linear fronts are real; early-time features are not

- **Real physics:** R(t) is close to linear after the transient in every
  run — a constant front speed, as expected for migration-controlled dense
  growth (same Chazalviel/Fleury picture as above). The overlay's slope
  ordering (dense runs steepest) matches the growth-rate figure.
- **Artifacts, must not be read as physics:**
  - The flat-topped dR/dt ≈ 90 µm/s at early times in 0.15/0.45/0.56 is the
    reporting continuity cap (3× the fastest measured front), i.e. the
    reported envelope catching up after batch reveals — not a real speed.
    Growth-law fits upstream use the uncapped radius, so the cap never
    enters a quoted number. Trust these runs' R(t) shape only past ~60 s.
  - The 0.45 % "stall" at t ≈ 40 s (R flat at ~3.8 mm) is the tail of that
    same reveal-then-catch-up sequence, not a growth pause.
  - dR/dt spikes (up to 50–90 µm/s in 0.02/0.04) are extremal-statistics
    noise: the enclosing radius only moves when the single outermost tip
    advances, so its derivative is intrinsically bursty and segmentation
    steps amplify it. Sustained levels, not bursts, carry meaning.
- Cosmetic caveat: viridis makes 0.02/0.04/0.06 nearly indistinguishable in
  the overlay; the per-panel tags disambiguate.

## 4a. Anatomy of the dR/dt features (per-panel read, R_and_dRdt_grid)

Every sharp feature in the derivative traces back to one of three causes:

1. **The identical flat-tops at exactly ~90 µm/s** (0.15/0.45/0.56 early
   frames) are the reporting **continuity cap** (VCAP = 3 × the fastest
   measured front, 30 µm/s): when segmentation first admits the young
   deposit as one batch (dense runs darken a sizeable disc at once, and the
   week-4 wire glare delays admission), the reported R must ramp from ~0 to
   millimetres at the maximum allowed slew rate. The giveaway that this is
   an artifact: the plateau height is numerically identical across runs.
2. **The sharp drop at the ramp's end** (and the brief R plateau right
   after, e.g. 0.45 % at t ≈ 40–55 s) is the reported R catching up to the
   true envelope and then waiting for it — a hand-off from cap-limited to
   physics-limited reporting, not a growth stall.
3. **Mid-run spikes over a low baseline** (all runs, clearest at
   0.02/0.04 %) are extremal statistics plus batch admissions: the
   enclosing radius moves only when the single outermost tip advances, so
   R(t) is a staircase and its derivative is intrinsically bursty;
   occasional segmentation admissions of a whole lobe (threshold crossing,
   glare clearing) add isolated spikes. The 4.5 s smoothing window shapes
   steps into bumps. dR/dt ≥ 0 always because the reported R is monotone by
   construction.

Sparse runs show no initial cap plateau because their enclosing radius
genuinely starts near zero and grows gradually — nothing to catch up to.
"R (raw)" (grey) is barely visible because it differs from the smoothed
curve only at staircase corners.

**Quantified ramp-leak check (2026-07-13):** refitting the growth rates
with cap-limited frames excluded moves 0.45 % from 32.0 → 30.3 µm/s (~5 %,
the largest shift — its ramp reaches into the fit window R ≥ 0.35 Rmax),
0.02 % from 14.2 → 13.5, and all others by ≤ 0.2 µm/s. Within the stated
systematics; quote 0.45 % as ~30–32 µm/s if pressed.

**Do the radii match the expected concentration dependence?** Compare
slopes and same-time radii, never final values: final R is set by run
duration (0.04 % ran 600 s, hence the largest final R at the slowest
front). At fixed t = 150 s the radii order as 4.3 / 4.4 / 5.2 / 5.5 / 7.0 /
6.6 mm for 0.02→0.56 % — monotone in concentration except the 0.45/0.56
inversion, which is inside run-to-run variability and carries the 0.45 %
lamp-heating confound. This matches the growth-rate figure and the
migration-limited expectation (weak, sub-linear speed increase).

## Bottom line vs predictions

The predicted picture — open DLA-like fractal at the dilute end, dense
compact growth (D → 2) above, faster fronts at higher concentration — is
what the figures show. The deviations that a skeptical reader will probe
(growth-rate non-monotonicities, the φ dip at 0.56 %, early-time dR/dt
plateaus, D orderings inside the dense band) are all either within stated
systematics or documented artifacts, and none supports an alternative
physical reading. The two caveats that belong IN the report text: the 0.45 %
lamp-heating confound, and that φ is not an independent check of D.
