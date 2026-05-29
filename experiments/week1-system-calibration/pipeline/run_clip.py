"""
run_clip.py  (pipeline)
-----------------------
Per-clip driver. Two phases, because curation is semi-automatic:

  phase 1 : detect -> link+drift -> curate -> contact sheet, then STOP.
            You review measurements/<run>/pipeline/sheet_*.png and confirm the
            clean set in labels.csv (keep = 1/0).
  phase 2 : MSD -> D, radius, aggregate -> k_B (reads labels.csv if present,
            else falls back to the auto proposal).

Usage (from the week root):
    python -m pipeline.run_clip run3 --phase 1
    # ...edit measurements/run3/pipeline/labels.csv...
    python -m pipeline.run_clip run3 --phase 2
    python -m pipeline.run_clip run3 --phase all        # 1 then 2 (auto labels)
"""

import argparse
import os

from . import paths, track, curate, contact_sheet, review, msd, radius, aggregate


def phase1(stem, videos_dir=None, max_frames=None, search=8, stub=50,
           sym_min=0.18, force=False):
    out = paths.out_dir(stem)
    tcsv = os.path.join(out, "trajectory.csv")
    if os.path.exists(tcsv) and not force:
        print(f"[run] trajectory.csv exists -> skipping tracking (use --force to redo)")
    else:
        track.run(stem, videos_dir=videos_dir, search=search, stub=stub,
                  max_frames=max_frames, sym_min=sym_min)
    curate.run(stem, videos_dir=videos_dir, max_frames=max_frames)
    contact_sheet.render(stem, videos_dir=videos_dir, max_frames=max_frames)
    try:
        review.render_all(stem)
    except Exception as e:                              # noqa: BLE001
        print(f"[run] review aids skipped ({e})")
    print(f"\n[run] PHASE 1 done. Review {out}\\sheet_singles.png + sheet_rejected.png,"
          f"\n      confirm labels.csv (keep=1/0), then: python -m pipeline.run_clip "
          f"{stem} --phase 2")


def phase2(stem, videos_dir=None, temp_C=25.0, delta_rho=60.0):
    msd.run(stem)
    radius.run(stem, videos_dir=videos_dir)
    return aggregate.run(stem, temp_C=temp_C, delta_rho=delta_rho,
                         videos_dir=videos_dir)


def main():
    ap = argparse.ArgumentParser(description="Per-clip Brownian pipeline driver.")
    ap.add_argument("run")
    ap.add_argument("--phase", choices=["1", "2", "all"], default="1")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--search", type=float, default=8)
    ap.add_argument("--stub", type=int, default=50)
    ap.add_argument("--sym-min", type=float, default=0.18)
    ap.add_argument("--temp-C", type=float, default=25.0)
    ap.add_argument("--delta-rho", type=float, default=60.0)
    ap.add_argument("--force", action="store_true", help="re-track even if trajectory.csv exists")
    ap.add_argument("--videos-dir", default=None)
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        phase1(args.run, videos_dir=args.videos_dir, max_frames=args.max_frames,
               search=args.search, stub=args.stub, sym_min=args.sym_min,
               force=args.force)
    if args.phase in ("2", "all"):
        phase2(args.run, videos_dir=args.videos_dir, temp_C=args.temp_C,
               delta_rho=args.delta_rho)


if __name__ == "__main__":
    main()
