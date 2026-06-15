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
