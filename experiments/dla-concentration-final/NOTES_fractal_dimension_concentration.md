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

The **defocused** runs (0.30 / 0.45 / 0.56 %) are **not reliably measurable** —
the mask captures a smooth blob, local D just reads ≈2. Excluded from any
quantitative claim.

These values agree with the original pipeline (0.02→1.61, dense→~1.89) to within
±0.05, so the interior-hole-fill artefact was real but small; it did **not**
manufacture the high dimensions.

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
