# Fractal dimension vs concentration — literature & method notes

Reference for the report Discussion. Explains the measured fractal dimensions,
the methodology, and why the dense runs plateau. Reflects the full audit +
two rounds of literature verification (2026-07-13).

## 1. The reliable result

Fractal dimension is only reliably measurable on the **focused** runs, where the
branch structure is optically resolved. Box-counting on the faithful mask (no
interior hole-filling), fit over the window [branch width, R/3], verified by a
window-stability sweep (D stable to ±0.05 as cutoffs vary):

| conc | D (reliable) | interpretation |
|---|---|---|
| 0.02 % | **1.69 ± 0.05** | open, diffusion-limited (DLA-like) fractal |
| 0.04 % | **1.91 ± 0.05** | dense / compact |
| 0.06 % | **1.87 ± 0.04** | dense / compact (cleanest plateau) |
| 0.15 % | **1.87 ± 0.06** | dense / compact |
| 0.45 % | **1.93 ± 0.07** | dense / compact (recovered, see §1a) |
| 0.56 % | **1.86 ± 0.06** | dense / compact (recovered, see §1a) |

The **defocused** runs (0.30 / 0.45 / 0.56 %) were re-audited (see
`NOTES_defocused_runs_recovery.md`): 0.56 % turned out to be *sharper than the
anchor* and gives a reliable **1.86 ± 0.06**; 0.45 % (mild ~1.6 px blur) gives
**1.93 ± 0.07** (validated window rule + Richardson–Lucy on a static median
stack); only 0.30 % remains **unrecoverable as a scaling exponent** (<0.7
decade after the blur cutoff; compact, D → 2; boundary D_b ≈ 1.3 ± 0.1 as a
distinct quantity).

These values agree with the original pipeline (0.02→1.61, dense→~1.89) to within
±0.05, so the interior-hole-fill artefact was real but small; it did **not**
manufacture the high dimensions.

## 1a. Physical soundness of the recovered 0.45 / 0.56 % values — and why they exceed the DLA 1.71

(Assessment logged 2026-07-13, after the recovery audit in
`NOTES_defocused_runs_recovery.md`.)

The values are physically sound, and exceeding 1.71 is the *expected*
signature that at these concentrations the growth is not diffusion-limited
aggregation at all. Witten–Sander's 1.71 belongs to a restrictive regime:
vanishingly dilute species, transport purely by diffusion, quasi-static
growth. The fractality of DLA comes from Laplacian **screening** (tips grow,
fjords starve). At 0.45–0.56 % CuSO₄ at 12 V that screening is destroyed:

1. **Screening-length collapse.** A DLA-like regime only exists between the
   branch width and the diffusion length ℓ ≈ D_ion/v_front. High
   concentration → plentiful ion supply, thin depletion layer, fast front →
   ℓ shrinks toward the branch scale. Below ℓ the interfacial flux is
   effectively uniform, neighbouring branches no longer starve each other,
   the front advances as a smooth envelope, and the object fills space —
   the dense-branching morphology, true D → 2 (Ben-Jacob 1986); ~1.9 is its
   finite-size reading.
2. **Electromigration.** At 12 V ion drift competes with diffusion;
   drift-dominated (quasi-ballistic) delivery also kills the screening
   instability. (Density *trend* remains concentration-driven per
   Sawada/Grier — migration explains non-diffusive transport, it is not a
   "voltage → dense" claim, which for Cu runs the other way, Kumar 2018.)
3. **Convection.** Stirring homogenises the concentration field. The 0.45 %
   run was lamp-heated throughout, guaranteeing thermal convection; its
   D_eff being the highest (1.93, brushing the ceiling) is consistent with
   that — but with ±0.07 overlapping 0.56 %'s 1.86, do NOT rank the two.
4. **Projection caveat (measurement-side).** The mask is a 2D projection of
   a finite-thickness deposit; projection can only fill gaps → a mild
   upward bias on top of the real physics.

**Consistency arguments:**
- 0.45 → 1.93 and 0.56 → 1.86 sit inside the dense bucket (0.04–0.15 % →
  1.87–1.91): the plateau extends across the whole dense range, as the
  D = 2 ceiling predicts. No new physics needed.
- **The series contains its own control:** the one genuinely
  diffusion-limited run (0.02 %) gives 1.69, matching Matsushita's 1.66 and
  ideal DLA 1.71. The method reproduces the DLA limit exactly where the
  physics is DLA — the strongest evidence that ~1.9 elsewhere is physical.
- With ~1 decade of window, 1.9 is indistinguishable from 2.0
  (Malcai/Avnir), so both runs are quoted as "compact, effectively
  space-filling, D → 2", not as fractals of dimension 1.86/1.93.
- Texture note: 0.56 % has notably finer branches (w ≈ 0.12 mm vs
  0.2–0.66 mm for 0.30–0.45 %), i.e. slightly more open structure inside
  the fit window — plausibly why it reads a touch lower. Interpret as
  texture within the compact family, not a step back toward the DLA branch.

## 2. Why the numbers make physical sense (morphology diagram)

Electrodeposition growth form is mapped against electrolyte concentration and
drive — the **morphology diagram** (Sawada, Dougherty & Gollub, *PRL* **56**,
1260, 1986; Grier, Ben-Jacob, Clarke & Sander, *PRL* **56**, 1264, 1986). Along
the **concentration** axis: low concentration → open **DLA-like fractal**
(D ≈ 1.6–1.7); higher concentration → **dense, compact** growth.

- **0.02 % → 1.69** sits on the open-fractal branch, matching the classic Zn-leaf
  electrodeposition value **D = 1.66 ± 0.03** (Matsushita et al., *PRL* **53**,
  286, 1984) and near 2D-DLA 1.71 (Witten & Sander, *PRL* **47**, 1400, 1981).
- **0.04–0.15 % → ~1.9** are dense-branching / compact deposits.

**Reframing that matters (verified):** dense-branching morphology is **not a
genuine fractal of dimension 1.9** — it is a *compact, homogeneous* structure
(branch width ≈ branch gap, smooth advancing envelope) whose true large-scale
dimension is the trivial **D → 2** (Ben-Jacob, Deutscher, Garik, Goldenfeld,
Lareah, *PRL* **57**, 1903, 1986). Measuring ~1.9 on a finite cluster is the
expected finite-size reading of a space-filling object. Quote it as
"compact, effectively space-filling, D → 2", **not** "a 1.9 fractal".

**Driver is concentration, NOT voltage.** Do not argue "12 V puts us in the dense
regime": Kumar et al. (*AIP Advances* **8**, 015219, 2018) find for Cu that *low*
voltage → dense branching and *high* voltage → fractal — the opposite direction.
Our density trend is set by the **concentration** axis (Sawada/Grier), and the
voltage argument is dropped.

Do **not** compare against Brady & Ball's D = 2.43 — that is a *3D* value.

## 3. Why the dense runs plateau

Two legitimate effects stack:
1. **The D = 2 ceiling (physics).** Dimension cannot exceed the embedding
   dimension (2 in the flat cell). As concentration densifies the growth, D
   climbs toward 2 and saturates; all dense runs pile up just under 2, so they
   cluster near 1.9. The transition 1.69 → 1.9 happens fast (0.02→0.04 %), then
   flattens. ("Plateau/saturation" is our descriptive language, not a literature
   term of art.)
2. **The <2-decade window (measurement).** With ~1 decade of scaling,
   distinguishing D = 1.9 from D = 2.0 on a finite cluster is genuinely marginal
   (Malcai, Lidar, Biham & Avnir, *Phys. Rev. E* **56**, 2817, 1997). So even if
   the dense deposits differ slightly, the method cannot resolve it — they read
   as a common ~1.9.

## 4. Why the methodology is correct

- **Box-counting** is the textbook estimator (Falconer, *Fractal Geometry*;
  Meakin, *Fractals, Scaling and Growth far from Equilibrium*, 1998).
- **Effective D over a limited window is normal**, provided the window and the
  linear scaling region are stated: most published experimental fractals rest on
  only **0.5–2 decades** (Malcai/Avnir 1997). Our ~1 decade is standard practice;
  we report an *effective* dimension, not an asymptotic one.
- **Faithful mask + window above the branch width.** Box sizes below the branch
  width measure solid material (slope → 2); the fractal information is at scales
  larger than a branch, so the fit belongs there. Interior hole-filling (kept for
  the enclosing-circle task) is removed for D.
- **Window-stability verified** — D holds to ±0.05 as the cutoffs move (inflating
  toward 2 only at R/2, the expected finite-size rollover).
- **Box-counting over sandbox for anisotropic clusters.** The sandbox/mass-radius
  estimator assumes radial symmetry about a chosen centre and is biased for
  anisotropic or off-centre clusters (Reyes-Ramírez et al., arXiv:1107.3336). The
  0.15 % deposit is a lopsided C-shape with a void, so its sandbox value (1.73)
  is unreliable; box-counting (no centre) gives the trustworthy 1.87. This is a
  justified local choice, not a claim that box-counting is universally superior.

## 5. The φ / D coupling (sanity-check figure)

The occupancy φ = M/πR² and D are **not independent**: for a fractal M ∝ R^D so
φ ∝ R^(D−2). Empirically corr(φ, D) = 0.994. φ is a consistency check, not
independent corroboration of D.

## 6. Bottom line for the report

- Report D as an **effective** box-counting dimension with the scaling window
  stated; reliable only for the four focused runs.
- **0.02 % is a genuine open DLA-like fractal (~1.7);** 0.04–0.15 % are
  **compact / effectively space-filling (D → 2, read as ~1.9 at finite size).**
- The plateau is the D = 2 ceiling plus limited resolving power; the driver is
  **concentration**, not voltage.
- Defocused runs are excluded as unmeasurable.

> Source caveat: some quantitative values (Kumar 2018; Wu 2023) are from
> abstracts/records (publisher full-texts blocked automated fetch). Verify exact
> numbers and the two 1986 PRL page locators against the originals before formal
> citation.
