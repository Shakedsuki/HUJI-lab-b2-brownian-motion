"""
probe_videos.py  (week1-system-calibration)
-------------------------------------------
Probe every clip in this week's videos/ for resolution, frame rate, frame count,
duration, and codec, then write videos_meta.json and print a grouped table.

WHY THIS MATTERS
  - Effective fps (avg_frame_rate = nb_frames / duration) is the TRUE time base.
    A camera asked for 9.3 fps often delivers less (exposure / USB bandwidth),
    and since the MSD t-axis is t = frame / fps, D scales linearly with it.
  - Resolution determines WHICH calibration applies. A µm/px measured on a
    1632x1224 ruler only transfers to 1632x1224 runs (unless the other mode is
    a pure AOI crop at the same magnification, which keeps pixel pitch fixed).

Uses ffprobe if on PATH (best); otherwise falls back to OpenCV.

Usage
-----
    cd experiments/week1-system-calibration
    python scripts/probe_videos.py                 # probes ./videos
    python scripts/probe_videos.py --dir some/dir
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import _paths


def _frac(s):
    """'4401/1000' -> 4.401 ; '30' -> 30.0 ; bad -> None."""
    try:
        if s in (None, "", "0/0", "N/A"):
            return None
        if "/" in s:
            n, d = s.split("/")
            d = float(d)
            return float(n) / d if d else None
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def probe_ffprobe(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = (j.get("streams") or [{}])[0]
    dur = _frac((j.get("format") or {}).get("duration"))
    nb = st.get("nb_frames")
    nb = int(nb) if nb and nb.isdigit() else None
    fps_avg = _frac(st.get("avg_frame_rate"))
    if nb is None and fps_avg and dur:
        nb = round(fps_avg * dur)
    return {
        "width": st.get("width"), "height": st.get("height"),
        "fps_nominal": _frac(st.get("r_frame_rate")),
        "fps_avg": fps_avg, "nb_frames": nb, "duration_s": dur,
        "codec": st.get("codec_name"),
    }


def probe_opencv(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()
    dur = (nb / fps) if (nb and fps) else None
    return {"width": w, "height": h, "fps_nominal": fps, "fps_avg": fps,
            "nb_frames": nb, "duration_s": dur, "codec": "unknown(cv2)"}


def main():
    ap = argparse.ArgumentParser(description="Probe video metadata.")
    ap.add_argument("--dir", default=_paths.VIDEOS_DIR)
    ap.add_argument("--out", default=os.path.join(_paths.WEEK_ROOT, "videos_meta.json"))
    args = ap.parse_args()

    have_ffprobe = shutil.which("ffprobe") is not None
    probe = probe_ffprobe if have_ffprobe else probe_opencv
    print(f"[probe] backend = {'ffprobe' if have_ffprobe else 'OpenCV (install ffmpeg for true avg fps)'}")

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith((".avi", ".mp4", ".tif", ".tiff")))
    if not files:
        sys.exit(f"No videos in {args.dir}")

    meta = {}
    print(f"\n{'file':32} {'WxH':>11} {'fps_avg':>8} {'fps_nom':>8} {'frames':>7} {'dur_s':>7} codec")
    print("-" * 90)
    for f in files:
        try:
            m = probe(os.path.join(args.dir, f))
        except Exception as e:
            print(f"{f:32} ERROR: {e}")
            continue
        if m is None:
            print(f"{f:32} (could not open)")
            continue
        meta[f] = m
        wxh = f"{m['width']}x{m['height']}"
        fa = f"{m['fps_avg']:.3f}" if m['fps_avg'] else "?"
        fn = f"{m['fps_nominal']:.3f}" if m['fps_nominal'] else "?"
        nb = m['nb_frames'] if m['nb_frames'] else "?"
        du = f"{m['duration_s']:.1f}" if m['duration_s'] else "?"
        print(f"{f:32} {wxh:>11} {fa:>8} {fn:>8} {str(nb):>7} {du:>7} {m['codec']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n[saved] {args.out}")

    # group by resolution so we know which calibration applies to which runs
    groups = {}
    for f, m in meta.items():
        groups.setdefault(f"{m['width']}x{m['height']}", []).append(f)
    print("\n[resolution groups]  (each group needs its own µm/px calibration)")
    for res, fs in sorted(groups.items()):
        print(f"  {res}: {len(fs)} files -> {', '.join(fs)}")


if __name__ == "__main__":
    main()
