"""
track_week2.py
--------------
Track week2 runs at downscale=2 (validated D-preserving on run3: median-D ratio
1.02, ~11x faster) -> writes measurements/<run>/pipeline/trajectory.csv each.
Idempotent: skips runs already tracked. The 11x speedup makes full 16-run
coverage cheap (~2.5 h), so we track everything and pick the best beads per
temperature downstream rather than pre-committing to a subset.

Usage:  python track_week2.py            # all runs in runs.json
        python track_week2.py run3 run6  # specific runs
"""
import os
import sys
import traceback

from pipeline import paths, track


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Track week2 runs (downscale=2 + multi-core FRST).")
    ap.add_argument("runs", nargs="*", help="run stems (default: all in runs.json)")
    ap.add_argument("--frames", type=int, default=None,
                    help="cap frames -> bounds wall-clock (1000 ~= ample for MSD)")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="detection processes (default cores-2)")
    ap.add_argument("--force", action="store_true",
                    help="re-track even if trajectory.csv exists")
    args = ap.parse_args()

    allruns = sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:]))
    only = args.runs or allruns
    print(f"[track_week2] {len(only)} run(s), downscale={args.downscale}, "
          f"workers={args.workers}"
          + (f", max_frames={args.frames}" if args.frames else "")
          + f": {only}", flush=True)
    for stem in only:
        tcsv = os.path.join(paths.out_dir(stem), "trajectory.csv")
        if os.path.exists(tcsv) and not args.force:
            print(f"[track_week2] {stem}: trajectory.csv exists -> skip (--force to redo)",
                  flush=True)
            continue
        print(f"\n===================== {stem} =====================", flush=True)
        try:
            track.run(stem, downscale=args.downscale, workers=args.workers,
                      max_frames=args.frames)
        except Exception:                       # noqa: BLE001
            traceback.print_exc()
            print(f"[track_week2] {stem} FAILED -- continuing", flush=True)
    print("\n[track_week2] DONE", flush=True)


if __name__ == "__main__":
    main()
