"""
finalize.py
-----------
Rebuild every week-2 k_B figure from the current on-disk state. Idempotent.
Run this AFTER process_all.py has finished (it re-runs msd, so don't run it
concurrently with a tracking batch that is still writing trajectory.csv) and
after any new radius tagging (python radius_tag.py <run>).

  1. Re-run the ENRICHED msd over every tracked run, so msd.csv + drift_perbead.csv
     carry sigma_D, R^2 and per-bead residual-drift velocities (+ sigma_v).
  2. Build the PRIMARY deliverable -- figures/kb_grid.png (per-run D-vs-1/r, slope
     = k_B; HAND-TAGGED radii, free diffusers only) + figures/kb_sweep_uniform.png
     + kb_grid_summary.csv -- via kb_grid.main(). Only runs with radius_manual.csv
     appear (tag more to fill the grid).
  3. Per-bead MSD + drift diagnostic grid (plot1_perbead.run) for the
     representative run per temperature (most free beads).

Usage:  python finalize.py
"""
import os
import sys
import traceback

import pandas as pd

from pipeline import paths
from pipeline import msd as msdmod
import kb_grid
import kb_summary
import plot1_perbead


def main():
    allruns = sorted(paths.load_runs().get("runs", {}), key=lambda s: int(s[3:]))

    # 1. enriched msd everywhere there is a trajectory
    print("[finalize] re-running enriched msd over all tracked runs...", flush=True)
    for stem in allruns:
        out = paths.out_dir(stem, make=False)
        if os.path.exists(os.path.join(out, "trajectory.csv")):
            try:
                msdmod.run(stem)
            except Exception:                       # noqa: BLE001
                traceback.print_exc()
                print(f"[finalize] msd {stem} FAILED -- continuing", flush=True)

    # 2. the grid + sweep + summary + headline printout
    print("\n[finalize] building kb_grid + sweep ...", flush=True)
    sys.argv = ["kb_grid.py"]                        # all analysed runs
    kb_grid.main()

    # synthesis figure: k_B/k_B^acc vs T with the near-ambient model (median est.)
    print("\n[finalize] building kb_summary (k_B vs T synthesis) ...", flush=True)
    sys.argv = ["kb_summary.py"]
    kb_summary.main()

    # clean publication grids: Fig 1 (MSD -> D) + Fig 2 (D vs 1/r -> k_B)
    print("\n[finalize] building plot1_publication (Fig 1: MSD) ...", flush=True)
    import plot1_publication
    plot1_publication.main()
    print("\n[finalize] building kb_grid_pub (Fig 2: publication grid) ...", flush=True)
    import kb_grid_pub
    kb_grid_pub.main()
    print("\n[finalize] building kb_strip (per-bead k_B by run headline) ...", flush=True)
    import kb_strip
    kb_strip.main()

    # 3. representative run per temperature (max free beads) -> per-bead figures
    summ = pd.read_csv(os.path.join(paths.FIGURES_DIR, "kb_grid_summary.csv"))
    reps = (summ.sort_values("n_free", ascending=False)
                .groupby("T_C", as_index=False).first())
    rep_runs = sorted(reps["run"], key=lambda s: int(s[3:]))
    print(f"\n[finalize] per-bead diagnostic figures for representatives "
          f"(max n_free per T): {rep_runs}", flush=True)
    for stem in rep_runs:
        try:
            plot1_perbead.run(stem)
        except Exception:                           # noqa: BLE001
            traceback.print_exc()
            print(f"[finalize] plot1_perbead {stem} FAILED -- continuing", flush=True)
    print("\n[finalize] DONE", flush=True)


if __name__ == "__main__":
    main()
