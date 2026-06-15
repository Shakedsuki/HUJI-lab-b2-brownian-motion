# Week 3 — DLA: Cu electrodeposition fractal dimension

Copper electrodeposition in a thin quasi-2D cell: CuSO₄ **0.29 %** solution,
**~12 V** between a central cathode wire and a copper ring anode. The deposit
grows by diffusion-limited aggregation (DLA) and its fractal dimension is
measured from photographs of the dried aggregate.

## Result

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

## Layout

```
media/     original camera photographs as shot (IMG_4123/4125/4127/4134 + CuSO4 cap label)
data/      analysis inputs (close-ups for measurement; dish shots + label for record)
scripts/   fractal_dimension.py — segmentation + box-counting + mass-radius
figures/   per-image diagnostics: segmentation overlay and both log-log fits
```

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

