"""
build_metadata.py  (week2-temperature-dependence)
=================================================
One-shot, re-runnable generator for this week's two metadata files. Keeping it
as a script (rather than hand-writing JSON) documents provenance: every value in
videos_meta.json comes straight from ffprobe, and every field in runs.json is
parsed from the acquisition filename -- nothing is fabricated.

  videos_meta.json : acquisition facts per clip (w, h, fps, frames, duration).
  runs.json        : per-run physics metadata. Temperatures come from the
                     FILENAME: T_start_C is the MEASURED starting temp (a real
                     thermometer reading, e.g. 24.3) and T_aim_C is the stage dial
                     SETPOINT. PROTOCOL (operator-confirmed): recording started
                     while the stage was still approaching the setpoint, but the
                     sample sat within ~+-1 C of T_start_C for the whole clip and
                     generally never reached T_aim_C. So the temperature used for
                     eta(T)/k_B is  T_C = T_start_C  with  T_unc_C = 1.0 C; T_aim_C
                     is kept for provenance only (NOT a physics input). window/roi
                     are null (full clip) until the QA pass proposes a settled,
                     clean sub-region.

Run:  python build_metadata.py        (from the week root)
"""

import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(HERE, "videos")

# Same optical mode as week1 (1632x1224 @ ~9.30 fps); the analyzable set.
FNAME_RE = re.compile(
    r"run(?P<n>\d+)_starting_temp_(?P<start>[\d.]+)C?_aiming_at_(?P<aim>[\d.]+)C",
    re.IGNORECASE)


def ffprobe(path):
    """Acquisition facts for one clip, straight from ffprobe."""
    def q(*entries):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", ",".join(entries), "-of", "json", path],
            capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    s = q("stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name")
    st = s["streams"][0]
    fmt = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path], capture_output=True, text=True, check=True)
    dur = float(json.loads(fmt.stdout)["format"]["duration"])

    def rate(r):
        a, b = r.split("/")
        return round(float(a) / float(b), 3)

    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps_nominal": rate(st["r_frame_rate"]),
        "fps_avg": rate(st["avg_frame_rate"]),
        "nb_frames": int(st["nb_frames"]),
        "duration_s": round(dur, 6),
        "codec": st["codec_name"],
    }


def parse_run(fname):
    m = FNAME_RE.search(fname)
    if not m:
        return None
    n = int(m.group("n"))
    start = float(m.group("start"))
    aim = float(m.group("aim"))
    return n, {
        "video": fname,
        "T_C": start,                # sample T during clip = MEASURED start temp
        "T_unc_C": 1.0,              # stayed within ~+-1 C of start over the clip
        "T_start_C": start,          # measured thermometer reading at record start
        "T_aim_C": aim,              # stage dial setpoint (NOT reached; provenance)
        "eta_cP": None,
        "radius_um": None,
        "window": None,              # [frame_start, frame_end]; null = full clip
        "roi": None,                 # [x0, y0, x1, y1]; null = full frame
        "status": "raw",
    }


def main():
    files = sorted(f for f in os.listdir(VIDEOS) if f.lower().endswith(".avi"))
    print(f"[scaffold] {len(files)} clips in {VIDEOS}")

    meta = {}
    runs = {}
    for f in files:
        meta[f] = ffprobe(os.path.join(VIDEOS, f))
        parsed = parse_run(f)
        if parsed is None:
            print(f"[scaffold] WARN: could not parse run/temps from {f!r}")
            continue
        n, rec = parsed
        runs[f"run{n}"] = rec
        m = meta[f]
        print(f"  run{n:<2} {m['width']}x{m['height']} @ {m['fps_avg']:.2f}fps  "
              f"{m['nb_frames']:>4} frames  start={rec['T_start_C']}C "
              f"aim={rec['T_aim_C']}C")

    # order runs.json numerically (run1, run2, ... run16)
    runs = {k: runs[k] for k in sorted(runs, key=lambda s: int(s[3:]))}

    with open(os.path.join(HERE, "videos_meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
        fp.write("\n")

    runs_doc = {
        "_schema": "per-run physics metadata; null = not yet measured. Do not "
                   "fabricate. T_start_C is the MEASURED starting temp; T_aim_C "
                   "is the stage setpoint. T_C (used for eta/k_B) stays null "
                   "until a measured steady/logged sample temperature is supplied "
                   "-- propagate T_band_C [aim,start] as a systematic until then. "
                   "window=[f0,f1], roi=[x0,y0,x1,y1] populated by the QA pass. "
                   "Acquisition facts live in videos_meta.json.",
        "_mode": "1632x1224 @ ~9.30 fps; same optics as week1 -> "
                 "calibration/scale.json (um_per_px=0.14381) transfers.",
        "runs": runs,
    }
    with open(os.path.join(HERE, "runs.json"), "w", encoding="utf-8") as fp:
        json.dump(runs_doc, fp, indent=2)
        fp.write("\n")

    print(f"[scaffold] wrote videos_meta.json ({len(meta)}) + runs.json "
          f"({len(runs)}) -> {HERE}")


if __name__ == "__main__":
    main()
