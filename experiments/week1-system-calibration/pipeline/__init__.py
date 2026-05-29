"""
pipeline/  (week1-system-calibration)
=====================================
A from-scratch, independent particle-tracking + radius pipeline for the
Brownian-motion k_B measurement. Designed to run in PARALLEL with the older
scripts/ pipeline (never clobbers it) so the two can be cross-checked.

Design choices (see the per-module docstrings for the why):
  * Detection: Fast Radial Symmetry Transform (polarity-free, circularity-
    preferring) instead of trackpy.locate, which mis-models the bright-field
    ring profile (bright/dark core flips through focus).
  * Linking: trackpy (the part that works).
  * Curation: purity-first, semi-automatic. The pipeline PROPOSES a clean
    singleton set + emits a scored contact sheet; the human confirms it into
    labels.csv. The decisive doublet killer is the temporal rigid-pair test.
  * Units: trajectories stay in PIXELS; physical conversion happens only at
    analysis (MSD/radius/aggregate), using calibration/scale.json (um/px) and
    each clip's MEASURED fps from videos_meta.json.

Run per-clip from the week root:
    python -m pipeline.run_clip run3 --phase 1   # detect -> curate -> sheet
    # ...confirm measurements/run3/pipeline/labels.csv...
    python -m pipeline.run_clip run3 --phase 2   # msd -> radius -> k_B
"""
