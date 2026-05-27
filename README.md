# HUJI Lab B2 — Brownian Motion

**Course:** Lab B2, 2026  
**Students:** Shaked Sukiennik, Nir Cohen  
**Instructor:** Shlomo Winberg  
**Experiment:** Optical tracking of Brownian motion of polystyrene microspheres in water; extraction of diffusion coefficient via MSD analysis and comparison to Stokes-Einstein prediction.

## Repository structure

```
data/
  raw/          # raw video files (.avi / .mp4) and any direct sensor output
  tracking/     # per-clip CSV outputs from the tracker (x, y, frame, t)
  processed/    # MSD tables, ensemble averages, derived quantities
scripts/
  tracking/     # particle detection & tracking
  analysis/     # MSD computation, diffusion coefficient fits, Stokes-Einstein
  figures/      # figure-generation scripts
report/
  lyx/          # LyX source files
  figures/      # final exported figures (PDF/PNG) for the report
notebooks/      # exploratory Jupyter notebooks (not report-final)
docs/           # reference PDFs, lab booklet, notes
```

## Key physics

- **MSD** `<r²(τ)> = 2d·D·τ` (d = spatial dimensions used; d=2 for 2D tracking)
- **Stokes-Einstein** `D = k_B T / (6πηr)` — links D to temperature T, solvent viscosity η, and particle radius r
- **Ergodic assumption:** time-averaged MSD ≈ ensemble-averaged MSD (assumed, not derived)
- **Stokes drag validity:** requires Re ≪ 1 and spherical particle far from walls

## Lab guide
https://einav.notion.site/Brownian-Motion-Lab-Guide-bd4ad5916db34151b70161b20dec0461
