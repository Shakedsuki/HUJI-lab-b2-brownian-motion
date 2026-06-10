"""
paths.py  (pipeline)
--------------------
Week-local path resolver for the v2 pipeline. Self-contained: resolves
everything relative to this package's week root, reuses the week's existing
JSON metadata READ-ONLY (calibration/scale.json, videos_meta.json, runs.json),
and writes outputs under measurements/<run>/pipeline/ so it never clobbers the
old scripts/ outputs in measurements/<run>/.

Worktree note
-------------
Raw videos are gitignored and live ONLY in the main checkout's videos/ folder.
When this package runs inside a git worktree (.../.claude/worktrees/<name>/...),
the local videos/ folder does not exist, so resolve_videos_dir() falls back to
the main checkout by stripping the ".claude/worktrees/<name>/" segment. An
explicit --videos-dir always wins.
"""

import os
import json

# pipeline/paths.py -> pipeline/ -> week root
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
WEEK_ROOT = os.path.dirname(PKG_DIR)

CALIB_DIR = os.path.join(WEEK_ROOT, "calibration")
MEAS_DIR = os.path.join(WEEK_ROOT, "measurements")
FIGURES_DIR = os.path.join(WEEK_ROOT, "figures")
RUNS_JSON = os.path.join(WEEK_ROOT, "runs.json")
SCALE_JSON = os.path.join(CALIB_DIR, "scale.json")
VIDEOS_META_JSON = os.path.join(WEEK_ROOT, "videos_meta.json")

OUT_TAG = "pipeline"   # measurements/<run>/pipeline/


def _strip_worktree(path):
    """Map a worktree path back to the main checkout by removing the
    '.claude/worktrees/<name>' segment. Returns path unchanged if not in a
    worktree."""
    parts = os.path.normpath(path).split(os.sep)
    try:
        i = parts.index(".claude")
        if parts[i + 1] == "worktrees":
            del parts[i:i + 3]                      # drop .claude/worktrees/<name>
            return os.sep.join(parts)
    except (ValueError, IndexError):
        pass
    return path


def resolve_videos_dir(override=None):
    """Directory holding the raw .avi clips (handles the worktree fallback)."""
    if override:
        return override
    local = os.path.join(WEEK_ROOT, "videos")
    if os.path.isdir(local) and any(f.endswith(".avi") for f in os.listdir(local)):
        return local
    main_videos = os.path.join(_strip_worktree(WEEK_ROOT), "videos")
    if os.path.isdir(main_videos):
        return main_videos
    raise FileNotFoundError(
        f"no videos/ with .avi found at {local!r} or {main_videos!r}; "
        f"pass --videos-dir explicitly")


def video(name, videos_dir=None):
    """Absolute path to a clip by name (pass-through if already a path)."""
    if os.path.isabs(name) or os.path.exists(name):
        return name
    return os.path.join(videos_dir or resolve_videos_dir(), name)


def out_dir(stem, make=True):
    """measurements/<stem>/pipeline/ (created on demand)."""
    d = os.path.join(MEAS_DIR, stem, OUT_TAG)
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def old_dir(stem):
    """measurements/<stem>/ — the OLD pipeline's outputs, for cross-checking."""
    return os.path.join(MEAS_DIR, stem)


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_scale_full():
    return _load(SCALE_JSON) or {}


def load_scale():
    """um_per_px (1632-mode), or None if unset."""
    return load_scale_full().get("um_per_px")


def load_runs():
    return _load(RUNS_JSON) or {}


def video_for_run(stem):
    """The .avi filename for a run stem from runs.json (fallback <stem>.avi)."""
    return load_runs().get("runs", {}).get(stem, {}).get("video", stem + ".avi")


def video_meta(name=None):
    m = _load(VIDEOS_META_JSON) or {}
    return m if name is None else m.get(os.path.basename(name))


def fps_of(name):
    m = video_meta(name)
    return m.get("fps_avg") if m else None


def dt_of(name):
    f = fps_of(name)
    return (1.0 / f) if f else None
