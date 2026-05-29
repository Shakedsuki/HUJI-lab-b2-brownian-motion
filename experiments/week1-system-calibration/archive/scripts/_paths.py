"""
_paths.py  (week1-system-calibration)
-------------------------------------
Week-LOCAL path resolver for a self-contained week. Each week owns its own
_paths.py and resolves everything RELATIVE TO ITS OWN FOLDER; nothing bleeds in,
nothing leaks out. Sibling scripts use it via `import _paths`.

Acquisition is NOT uniform this week: there are two camera modes
  - 1632x1224 @ ~9.30 fps  -> run2..run10 + both calibration videos (the real set)
  - 2560x1920 @ ~4.40 fps  -> run1, test (exploratory; no ruler at this res)
so fps/resolution are per-video (read from videos_meta.json), not global constants.
"""

import os
import json

# scripts/_paths.py -> scripts/ -> week root
WEEK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEOS_DIR       = os.path.join(WEEK_ROOT, "videos")
CALIB_DIR        = os.path.join(WEEK_ROOT, "calibration")
MEAS_DIR         = os.path.join(WEEK_ROOT, "measurements")
FIGURES_DIR      = os.path.join(WEEK_ROOT, "figures")
RUNS_JSON        = os.path.join(WEEK_ROOT, "runs.json")
SCALE_JSON       = os.path.join(CALIB_DIR, "scale.json")
VIDEOS_META_JSON = os.path.join(WEEK_ROOT, "videos_meta.json")


def video(name):
    """Absolute path to a clip in videos/ by name (pass-through if already a path)."""
    return name if os.path.isabs(name) or os.path.exists(name) else os.path.join(VIDEOS_DIR, name)


def clip_dir(stem):
    """measurements/<stem>/ for this week (callers create on demand)."""
    return os.path.join(MEAS_DIR, stem)


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_scale():
    """um_per_px from calibration/scale.json (1632-mode), or None if unset."""
    j = _load(SCALE_JSON)
    return j.get("um_per_px") if j else None


def load_runs():
    return _load(RUNS_JSON) or {}


def video_meta(name=None):
    """Full videos_meta.json, or the entry for one file (basename)."""
    m = _load(VIDEOS_META_JSON) or {}
    if name is None:
        return m
    return m.get(os.path.basename(name))


def fps_of(name):
    m = video_meta(name)
    return m.get("fps_avg") if m else None


def dt_of(name):
    """Seconds per frame for a given clip, from its measured average fps."""
    f = fps_of(name)
    return (1.0 / f) if f else None
