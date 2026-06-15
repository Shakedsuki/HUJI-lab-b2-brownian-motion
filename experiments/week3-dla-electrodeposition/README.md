# Week 3 — DLA: Cu electrodeposition (fractal dimension + growth kinetics)

Copper electrodeposition in a thin quasi-2D cell: CuSO₄ **0.29 %** solution,
**~12 V** between a central cathode wire and a copper ring anode. The deposit
grows by diffusion-limited aggregation (DLA). Its **fractal dimension** is
measured from photographs of the dried aggregate (below), and its **growth
kinetics** R(t), M(t) are measured from videos of the deposition
([Growth kinetics from video](#growth-kinetics-from-video--rt-mt)).

## Result — fractal dimension (still photos)

```
D = 1.65 ± 0.04      (DLA theory in 2D: D ≈ 1.71)
```

Mean over three estimates (box-counting on both close-ups + mass-radius on
IMG_4125); the error combines the spread between estimates with a ±0.03
systematic from the segmentation-threshold scan (0.85 ± 0.05).

| image | method | D |
|---|---|---|
| IMG_4125 | box counting | 1.673 ± 0.018 |
| IMG_4125 | mass-radius | 1.628 ± 0.013 |
| IMG_4127 | box counting | 1.663 ± 0.018 |
| IMG_4127 | mass-radius | (1.91 — excluded: wire occludes the centre, oblique angle) |

At ~12 V the growth is not purely diffusion-limited — the strong field adds a
drift component, which is known to push the deposit toward denser, more radial
branches; a slight D below or near the ideal 1.71 is expected.

## Growth kinetics from video — R(t), M(t)

The still photos above fix the fractal dimension of the *final* deposit but carry
no time axis. Seven videos of the deposition (1280×720, 59.94 fps) add it:
tracking the aggregate frame-by-frame gives the DLA growth laws

```
M ∝ Rg^D      D = 1.6 ± 0.1     (run 4: 1.72 ± 0.08,  DSC_0076: 1.56 ± 0.03)
M ∝ t^α       α ≈ 0.85–0.9      (≈ constant-current / Faradaic deposition)
Rg ∝ t^β      β ≈ 0.52          (≈ 1/D — the DLA radius law  R ∝ t^(1/D))
```

measured on the two **close-up** runs (run 4, DSC_0076), where individual
branches are resolved. The mass–radius exponent **D = 1.6 ± 0.1 measured from the
moving deposit independently reproduces the static D = 1.65 ± 0.04** and the
2D-DLA value ≈ 1.71 — a cross-check from a wholly different measurement. The three
laws close self-consistently: **D ≈ α/β** (1.73 and 1.63 for the two clips), which
would fail for a mis-chosen nucleation time and is the internal validation.

| clip | view | α (M∝t) | β (Rg∝t) | D (M∝Rgᴰ) | α/β |
|---|---|---|---|---|---|
| run 4 (dense) | close-up | 0.89 ± 0.06 | 0.52 ± 0.01 | **1.72 ± 0.08** | 1.73 |
| DSC_0076 (radial) | close-up | 0.85 ± 0.01 | 0.52 ± 0.01 | **1.56 ± 0.03** | 1.63 |
| run 1 / 2 / 3 | wide | — | — | (2.0–3.0, resolution-limited) | — |

**Physics.** α ≈ 0.9 ≈ 1: the deposited mass grows roughly linearly in time, i.e.
approximately constant-current (Faradaic) deposition, with a mild slowdown as the
cell depletes. β ≈ 0.52 ≈ 1/D is the DLA expectation that a cluster fed at a
constant rate spreads as R ∝ t^(1/D). In **run 3** the voltage was lowered
12 → 5–6 V partway through, and Rg(t) and M(t) visibly roll over to a plateau as
the current drops — a direct illustration of the current-dependence of the growth
(its wide-view D is resolution-limited and not quoted).

**Why only the close-ups give D.** In the wide-dish views the whole deposit spans
only ~100–270 px, so the 1–2 px branches and the faint inter-branch shading blur
into a filled footprint and the apparent dimension rises toward 2 (run 1/2/3 give
2.0–3.0). The close-ups resolve the branches, so their M–Rg scaling is the
trustworthy fractal measure — visible directly in `figures/kinetics_summary.png`,
where only the two close-ups trace a clean, wide-range power law.

## Layout

```
media/     original camera photographs as shot (IMG_4123/4125/4127/4134 + CuSO4 cap label)
data/      analysis inputs (close-ups for measurement; dish shots + label for record)
           + kinetics_<clip>.csv  (per-frame t, M, R95, R99.5, Rg, n_comp)
scripts/   fractal_dimension.py  — segmentation + box-counting + mass-radius (stills)
           growth_kinetics.py    — per-frame R(t)/M(t) tracking + DLA growth-law fits
figures/   fractal_<image>.png   — per-image fractal-dimension diagnostics
           kinetics_<clip>.png   — per-clip kinetics (segmented frames + the three fits)
           kinetics_summary.png  — Rg(t) and M~Rg for all clips together
```

The deposition videos (~0.5 GB each) live outside the repo; point the kinetics
script at them with the `WEEK3_VIDEO_DIR` environment variable.

## Method (scripts/fractal_dimension.py)

1. **Segmentation** — flat-field the grayscale (divide by a σ=101 px Gaussian
   blur), threshold dark pixels at 0.85, mask the green/yellow electrode wire
   in HSV, and drop speckle outside the aggregate disc.
2. **Box counting** — N(s) ~ s⁻ᴰ, fitted between the branch width (~8 px) and
   R/8, where the local slope is scale-free.
3. **Mass-radius** — M(<r) ~ rᴰ about the aggregate centroid, each annulus
   corrected for the wire-occluded area; auto-flagged unreliable when the wire
   covers >25 % of the inner quarter-disc (the IMG_4127 case).

Run: `python3 scripts/fractal_dimension.py`

## Kinetics method (scripts/growth_kinetics.py)

Per frame, the *growing* deposit is isolated by exploiting that it is the only
thing in the cell that changes:

1. **Temporal background subtraction** — reference = median of the first frames;
   the dish, copper-ring anode, wire, reflections, dust, and any leftover deposit
   from a previous run are static and cancel, so only new growth survives. This
   alone removes the stray central blob present in run 2 / run 3.
2. **Flat-field branch tracing** — within the changed region, dark dendrite pixels
   are kept by the same flat-field local-contrast threshold as the static analysis
   (divide by a σ=101 px blur), run as a hysteresis so faint tips are followed
   *without* flood-filling the smooth depletion halo into a solid blob.
3. **Wire removal** — the green/yellow cathode wire (the only moving coloured
   object) is segmented in HSV each frame and excluded wherever it currently is.
4. **Cluster gate** — only the component connected to the cathode-tip seed (the
   persistent early deposit closest to the wire) is kept, dropping far dust.

Per frame we record mass *M* (deposit area, px), radius of gyration *Rg*, the
95th-percentile reach *R95* and *Rmax*. Exponents are fit over an **objective
growth window** — mass between 8 % and 75 % of the plateau, past the nucleation
lag and before the current-starved saturation — with time measured from the
observed nucleation *t0*. The radius used is **Rg**: the reach *R95* saturates
once branches approach the frame edge, which drives the apparent D above 2 (the
unphysical value is the tell). Quoted errors combine the statistical slope error
with a systematic from varying the fit window (which dominates). *R* is in pixels:
the exponents are scale-free, so no px→mm calibration is needed.

Run: `WEEK3_VIDEO_DIR=/path/to/videos python3 scripts/growth_kinetics.py`
(decodes with ffmpeg; writes `data/kinetics_<clip>.csv` and the figures).

## Kinetics figure caption (`figures/kinetics_<clip>.png`)

> **Growth kinetics of the copper electrodeposit.** *(Top)* three segmented
> frames (early / mid / late) with the extracted deposit in **green**, the
> cathode-tip seed (dot) and the *R95* circle. *(Bottom-left)* radius of gyration
> *Rg* and reach *R95* (left axis) and mass *M* (right axis) versus time, with the
> fit window highlighted. *(Bottom-centre)* the radius law *Rg ∝ (t − t0)^β* on
> log-log axes. *(Bottom-right)* the mass–radius relation *M ∝ Rg^D*, whose slope
> is the fractal dimension; the inset gives α (from *M ∝ t^α*) and the consistency
> ratio α/β, which equals *D* for self-similar growth.

## Figure caption (`figures/fractal_<image>.png`)

> **Fractal-dimension analysis of the copper electrodeposit (CuSO₄ 0.29 %, ~12 V).**
> Each diagnostics figure has three panels.
>
> **(Left) Segmentation.** The original photograph with the extracted deposit
> overlaid in **red** and the masked electrode wire in **blue**; the dashed
> **yellow** circle marks the aggregate radius *R* (centroid + 99th-percentile
> pixel radius). The deposit is isolated by flat-fielding the grayscale image
> — dividing it by a σ = 101 px Gaussian blur to remove the uneven paper/dish
> illumination — then keeping pixels darker than 0.85× the local background.
> The saturated green/yellow wire is identified in HSV and dilated by 25 px so
> its halo is excluded, and connected components smaller than 5 px or lying
> outside 1.1 *R* are discarded as speckle. This panel is the visual check that
> we measured the dendrite and *only* the dendrite.
>
> **(Centre) Box-counting dimension.** The segmented mask is tiled with square
> boxes of side *s*, and N(*s*), the number of boxes containing any deposit, is
> plotted against *s* on log-log axes. A fractal obeys N(*s*) ∝ s⁻ᴰ, so the
> slope is −*D*. **Grey** points span all box sizes; **red** points are the
> fit window (8 px ≤ *s* ≤ *R*/8), and the black line is the least-squares fit
> whose slope gives the quoted *D*. Points below ~8 px flatten toward slope 2
> (boxes resolve the solid interior of a branch) and points above *R*/8 steepen
> toward the finite size of the cluster; both are excluded so the fit sees only
> the scale-free regime.
>
> **(Right) Mass-radius dimension.** The cumulative deposited "mass" M(<*r*)
> — the number of deposit pixels within radius *r* of the centroid — versus *r*
> on log-log axes. A fractal obeys M(<*r*) ∝ rᴰ, so here the slope *is* +*D*.
> Mass is accumulated in geometric annuli, and each annulus is corrected for
> the area hidden behind the wire (mass scaled by total/visible area); annuli
> more than 60 % occluded are dropped. **Grey** points are all radii, **blue**
> points the fit window (30 px ≤ *r* ≤ 0.8 *R*, i.e. outside the central
> electrode blob and inside the under-grown rim), and the black line is the
> fit. This panel is auto-flagged *unreliable* when the wire covers >25 % of
> the inner quarter-disc, since the centre mass is then unmeasurable and the
> slope is biased high — the case for IMG_4127.
>
> The agreement between the independent box-counting and mass-radius slopes
> (and across the two images) is what justifies quoting a single
> **D = 1.65 ± 0.04**, consistent with the 2D DLA value of ≈ 1.71.

