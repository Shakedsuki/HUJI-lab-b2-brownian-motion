"""
process_all.py
--------------
Drive the full AUTOMATIC week-2 pipeline over every run, uniformly, so the
temperature sweep is an apples-to-apples comparison (same detector, same
curation, same outer-edge radius -- exactly the week-1 method, applied per run at
each run's MEASURED starting temperature).

Per run, idempotently:
  1. track   (downscale=2 FRST + trackpy)         -> trajectory.csv   [skip if present]
  2. curate  (auto singleton proposal)            -> curation_proposed.csv
  3. msd     (per-bead D from MSD = 4D tau + c)    -> msd.csv
  4. radius  (outer-edge ring fit, AUTO)           -> radius.csv

No human-in-the-loop tagging is used here: the grid is built from the AUTO
curation + AUTO radius for ALL runs so no run is special. (run7/run15 also carry
a separate manual-radius refinement; that is reported only as a cross-check.)

Progress is line-buffered to stdout so it can be tail/Monitor-ed.

Usage:  python process_all.py                 # all runs in runs.json
        python process_all.py run3 run4       # specific runs
        python process_all.py --frames 1500   # cap frames (bound wall-clock)
"""
import argparse
import os
import sys
import time
import traceback

from pipeline import paths, track, curate, msd, radius


def process(stem, downscale, workers, frames, force_track=False):
    out = paths.out_dir(stem)
    tcsv = os.path.join(out, "trajectory.csv")
    t0 = time.time()

    if force_track or not os.path.exists(tcsv):
        print(f"[{stem}] TRACK (downscale={downscale}, workers={workers}, "
              f"frames={frames or 'all'})...", flush=True)
        track.run(stem, downscale=downscale, workers=workers, max_frames=frames)
    else:
        print(f"[{stem}] trajectory.csv present -> skip track", flush=True)

    print(f"[{stem}] CURATE...", flush=True)
    curate.run(stem, max_frames=frames)
    print(f"[{stem}] MSD...", flush=True)
    msd.run(stem)
    print(f"[{stem}] RADIUS...", flush=True)
    radius.run(stem)
    print(f"[{stem}] DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Automatic week-2 pipeline over all runs.")
    ap.add_argument("runs", nargs="*", help="run stems (default: all in runs.json)")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--frames", type=int, default=1500,
                    help="cap frames per clip (ample for MSD; bounds wall-clock). "
                         "Use 0 for the full clip.")
    ap.add_argument("--force-track", action="store_true")
    args = ap.parse_args()
    frames = args.frames or None

    allruns = sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:]))
    only = args.runs or allruns
    print(f"[process_all] {len(only)} run(s): {only}", flush=True)
    ok, bad = [], []
    for stem in only:
        try:
            process(stem, args.downscale, args.workers, frames, args.force_track)
            ok.append(stem)
        except Exception:                       # noqa: BLE001
            traceback.print_exc()
            print(f"[process_all] {stem} FAILED -- continuing", flush=True)
            bad.append(stem)
    print(f"\n[process_all] DONE. ok={ok}  failed={bad}", flush=True)


if __name__ == "__main__":
    main()
