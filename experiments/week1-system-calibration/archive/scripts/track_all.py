"""
track_all.py  (week1-system-calibration)
----------------------------------------
Batch driver: run track.py over several clips with one set of parameters,
sequentially, skipping any run that already has a trajectory.csv (so it resumes
cleanly if interrupted). Intended for the room runs (run2-run6), which share
acquisition conditions and therefore the same diameter/minmass tuned on run2.

The temperature runs (run7-run10) may have shifted illumination/contrast, so
tune-check one of them with `track.py --tune` before adding them here.

Usage
-----
    cd experiments/week1-system-calibration
    # room set, run2's parameters:
    python scripts/track_all.py --runs run3 run4 run5 run6 \
        --diameter 11 --minmass 200 --search 15 --memory 3
    # default --runs is run2..run6
    python scripts/track_all.py --diameter 11 --minmass 200
"""

import argparse
import os
import subprocess
import sys
import time

import _paths

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.join(HERE, "track.py")


def main():
    ap = argparse.ArgumentParser(description="Batch-track several clips with track.py.")
    ap.add_argument("--runs", nargs="+",
                    default=["run2", "run3", "run4", "run5", "run6"],
                    help="run stems to track")
    ap.add_argument("--diameter", type=int, default=11)
    ap.add_argument("--minmass", type=float, default=200)
    ap.add_argument("--search", type=float, default=15)
    ap.add_argument("--memory", type=int, default=3)
    ap.add_argument("--stub", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="re-track even if trajectory.csv exists")
    args = ap.parse_args()

    done, skipped, failed = [], [], []
    for run in args.runs:
        traj = os.path.join(_paths.clip_dir(run), "trajectory.csv")
        if os.path.exists(traj) and not args.force:
            print(f"[skip] {run}: trajectory.csv already exists")
            skipped.append(run)
            continue

        video = run if run.endswith(".avi") else run + ".avi"
        if not os.path.exists(_paths.video(video)):
            print(f"[miss] {run}: no video {video} in videos/ -- skipping")
            failed.append(run)
            continue

        cmd = [sys.executable, TRACK, video,
               "--diameter", str(args.diameter),
               "--minmass", str(args.minmass),
               "--search", str(args.search),
               "--memory", str(args.memory),
               "--stub", str(args.stub)]
        print(f"\n{'='*60}\n[run] {run}: {' '.join(cmd[2:])}\n{'='*60}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=os.path.dirname(HERE))  # cwd = week root
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"[done] {run} in {dt:.0f}s")
            done.append(run)
        else:
            print(f"[FAIL] {run} (exit {r.returncode})")
            failed.append(run)

    print(f"\n{'='*60}\nsummary: tracked {done}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
