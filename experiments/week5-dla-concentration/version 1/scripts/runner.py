import sys, csv, subprocess
from pathlib import Path
CAP = 80
N = dict(run1=654, run2=976, run3=695)
parts = Path("/tmp/w5/parts"); parts.mkdir(exist_ok=True)
tmp = Path("/tmp/w5")
procs = []
for tag, n in N.items():
    have = set()
    for p in parts.glob(f"{tag}_*.csv"):
        with open(p) as fh:
            for row in csv.reader(fh):
                if row: have.add(int(row[0]))
    k0 = 0
    while k0 in have: k0 += 1
    if k0 >= n: continue
    if k0 > 0 and not (tmp / f"state_{tag}_{k0:05d}.npz").exists(): continue
    if len(procs) >= 1: break
    procs.append(subprocess.Popen(
        [sys.executable, "week5_analysis.py", "--stage", "measure",
         "--runs", tag, "--k0", str(k0), "--k1", str(min(k0+CAP, n))],
        cwd="/tmp/wk/scripts"))
for p in procs: p.wait()
print("launched", len(procs))
