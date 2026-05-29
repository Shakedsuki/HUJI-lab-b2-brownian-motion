"""
frames.py  (pipeline)
---------------------
Streaming frame I/O + temporal-median flat-field.

Two access patterns, both indexing frames from 0 in decode order (consistent
for these intra-only MJPEG AVIs):
  * iter_frames(path)  : sequential stream, ONE frame in RAM at a time (the long
                         clips are ~1200-1900 frames of 1632x1224 -> a full list
                         would be GBs). Used by detection.
  * frame_at(path, i)  : random access via OpenCV seek (MJPEG is all-intra, so
                         seeks are exact + fast). Used for per-bead crops and
                         flat-field sampling.

flat_field(): temporal median over evenly-spaced frames. Moving beads average
out; what remains is static illumination + fixed debris + the big out-of-focus
halos that don't move. Subtracting it flattens the field and kills those halos
before detection.
"""

import os
import sys
import numpy as np


def _gray(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    return a.astype(np.float32)


def iter_frames(path, max_frames=None):
    """Yield grayscale float32 frames (imageio/pyav; OpenCV fallback)."""
    try:
        import imageio.v3 as iio
        for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
            if max_frames is not None and i >= max_frames:
                return
            yield _gray(fr)
        return
    except Exception as e:                              # noqa: BLE001
        print(f"[frames] imageio/pyav failed ({e}); using OpenCV", file=sys.stderr)
    import cv2
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames is not None and i >= max_frames):
            break
        yield _gray(fr)
        i += 1
    cap.release()


def count_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def frame_at(path, idx):
    """Random-access a single grayscale float32 frame, or None."""
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, fr = cap.read()
    cap.release()
    return _gray(fr) if ok else None


def get_flat(path, cache_path=None, n_sample=60, max_frames=None):
    """Flat-field with on-disk caching, so stages (curate/sheet/review/radius)
    don't each rebuild it (60 frame reads) from scratch."""
    if cache_path and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:                               # noqa: BLE001
            pass
    flat = flat_field(path, n_sample=n_sample, max_frames=max_frames)
    if cache_path:
        try:
            np.save(cache_path, flat)
        except Exception:                               # noqa: BLE001
            pass
    return flat


def flat_field(path, n_sample=60, max_frames=None):
    """Temporal median over n_sample evenly-spaced frames -> static background."""
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames) if total > 0 else max_frames
    if total <= 0:
        total = n_sample
    idxs = np.unique(np.linspace(0, max(total - 1, 0),
                                 min(n_sample, total)).astype(int))
    picked = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            picked.append(_gray(fr))
    cap.release()
    if not picked:
        raise RuntimeError(f"flat_field: could not read any frames from {path}")
    return np.median(np.stack(picked), axis=0).astype(np.float32)


if __name__ == "__main__":   # quick smoke test: python -m pipeline.frames <run>
    import argparse
    from . import paths
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="run3")
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()
    vid = paths.video(paths.video_for_run(args.run), args.videos_dir)
    print(f"video: {vid}")
    print(f"cv2 frame count: {count_frames(vid)}")
    print(f"measured fps: {paths.fps_of(paths.video_for_run(args.run))}")
    f0 = frame_at(vid, 0)
    print(f"frame0 shape={f0.shape} dtype={f0.dtype} "
          f"min={f0.min():.1f} max={f0.max():.1f} mean={f0.mean():.1f}")
    ff = flat_field(vid, n_sample=40)
    print(f"flat-field shape={ff.shape} mean={ff.mean():.1f} "
          f"(residual std after subtract on frame0: {(f0 - ff).std():.2f})")
