"""
review_widget.py <run>
----------------------
Human-in-the-loop curation review. After curate.run, this surfaces the beads
that need a human eye -- the auto-proposed singles (to confirm) plus the
BORDERLINE rejects (single soft reason) -- as a self-contained clickable HTML
grid: each bead a crop + its metrics, click to toggle keep/reject, Submit calls
sendPrompt with the final keep-list. The assistant turns that verdict into
labels.csv and re-runs MSD/radius.

Confident multi-reason rejects are auto-dropped (not shown); confident singles
are shown pre-selected green so you only spend attention on the doubtful ones.

Writes measurements/<run>/pipeline/review_<run>.html (the assistant relays it via
the visualize widget). Usage:  python review_widget.py run7
"""
import base64
import os
import sys

import numpy as np
import pandas as pd

from pipeline import paths, frames as fr

HARD = {"rigid-doublet", "two-cores", "mislink"}     # never borderline


def _classify(row):
    reasons = [r for r in str(row["reason"]).split(";") if r]
    if row["proposed"] == "single":
        return "keep"
    if len(reasons) == 1:
        return "drop" if reasons[0] in HARD else "borderline"
    return "drop"


def _crop_b64(img, x, y, r, out_px=44):
    """Square crop around a bead, contrast-normalised, resized to out_px so the
    whole grid stays small enough to ship inline to the visualize widget."""
    import cv2
    h = int(max(2.2 * r, 16))
    H, W = img.shape
    a, b = max(0, int(y) - h), max(0, int(x) - h)
    c = img[a:min(H, int(y) + h), b:min(W, int(x) + h)]
    if c.size == 0:
        return None
    c = c - c.min()
    c = (c / (c.max() + 1e-9) * 255).astype(np.uint8)
    c = cv2.resize(c, (out_px, out_px), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", c)
    return base64.b64encode(buf).decode() if ok else None


def build(stem, max_show=28):
    import cv2
    out = paths.out_dir(stem)
    cur = pd.read_csv(os.path.join(out, "curation.csv"))
    traj = pd.read_csv(os.path.join(out, "trajectory.csv"))
    vid = paths.video(paths.video_for_run(stem))
    flat = fr.get_flat(vid, cache_path=os.path.join(out, "flat.npy"))

    cur["cls"] = cur.apply(_classify, axis=1)
    show = cur[cur["cls"].isin(["keep", "borderline"])].copy()
    # borderline first (most need attention), then strongest singles
    show = show.sort_values(["cls", "sym_med"], ascending=[True, False]).head(max_show)

    mid = {int(p): g.sort_values("frame").iloc[len(g) // 2]
           for p, g in traj.groupby("particle")}
    cap = cv2.VideoCapture(vid)
    items = []
    for _, r in show.iterrows():
        pid = int(r["particle"])
        if pid not in mid:
            continue
        m = mid[pid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(m["frame"]))
        ok, frm = cap.read()
        if not ok:
            continue
        img = frm[..., :3].mean(-1).astype(np.float32) - flat
        b64 = _crop_b64(img, m["x_raw"], m["y_raw"], r["R_px_med"])
        if not b64:
            continue
        items.append(dict(
            pid=pid, b64=b64, cls=r["cls"],
            sym=round(float(r["sym_med"]), 2), ecc=round(float(r["ecc_med"]), 2),
            resid=round(float(r["resid_med"]), 3), R=round(float(r["R_px_med"]), 1),
            n=int(r["n_frames"]), reason=(str(r["reason"]) or "clean single")))
    cap.release()

    p = os.path.join(out, f"review_{stem}.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_html(stem, items))
    nb = sum(i["cls"] == "borderline" for i in items)
    print(f"[review] {stem}: {len(items)} beads ({len(items)-nb} auto-keep to "
          f"confirm, {nb} borderline) -> {p}")
    return p


def _html(stem, items):
    cells = []
    for it in items:
        sel = "keep" if it["cls"] == "keep" else "reject"
        tag = "single" if it["cls"] == "keep" else it["reason"]
        cells.append(
            f'<div class="cell {sel}" data-pid="{it["pid"]}" onclick="tog(this)">'
            f'<img src="data:image/png;base64,{it["b64"]}"/>'
            f'<div class="m">p{it["pid"]} · n{it["n"]}<br>sym {it["sym"]} · '
            f'ecc {it["ecc"]}<br><span class="r">{tag}</span></div></div>')
    grid = "".join(cells)
    return f"""<style>
 .rvwrap{{font-family:var(--font-sans);color:var(--color-text-primary);padding:1rem 0}}
 .rvbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding-bottom:10px;
   margin-bottom:12px;border-bottom:0.5px solid var(--color-border-tertiary)}}
 .rvgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(92px,1fr));gap:10px}}
 .cell{{border:2px solid var(--color-border-tertiary);border-radius:var(--border-radius-md);
   padding:5px;cursor:pointer;text-align:center;background:var(--color-background-secondary)}}
 .cell img{{width:100%;aspect-ratio:1;object-fit:contain;image-rendering:pixelated;
   background:#000;border-radius:4px;display:block}}
 .cell.keep{{border-color:var(--color-border-success)}}
 .cell.reject{{border-color:var(--color-border-danger);opacity:.5}}
 .m{{font-size:11px;margin-top:4px;line-height:1.3;color:var(--color-text-secondary)}}
 .rs{{color:var(--color-text-tertiary)}}
 .cnt{{font-size:13px;font-weight:500}}
</style>
<div class="rvwrap">
 <h2 class="sr-only">Curation review grid for {stem}: click each bead crop to keep or reject it.</h2>
 <div class="rvbar">
  <span style="font-size:13px;color:var(--color-text-secondary)">Click a bead to toggle.
   green = keep, red = reject. Reject doublets, debris, blurry; keep round singles.</span>
  <button onclick="rall('keep')">All keep</button>
  <button onclick="rall('reject')">All reject</button>
  <span class="cnt" id="rvcnt"></span>
  <button onclick="rgo()" style="border-color:var(--color-border-success)">Submit verdict ↗</button>
 </div>
 <div class="rvgrid">{grid}</div>
</div>
<script>
 function rupd(){{var k=document.querySelectorAll('.cell.keep').length,
   t=document.querySelectorAll('.cell').length;
   document.getElementById('rvcnt').textContent=k+' / '+t+' kept';}}
 function tog(e){{e.classList.toggle('keep');e.classList.toggle('reject');rupd();}}
 function rall(c){{document.querySelectorAll('.cell').forEach(function(e){{
   e.classList.remove('keep','reject');e.classList.add(c);}});rupd();}}
 function rgo(){{var k=[].slice.call(document.querySelectorAll('.cell.keep'))
   .map(function(e){{return e.dataset.pid;}});
   sendPrompt('CURATION-VERDICT {stem} KEEP=['+k.join(',')+']');}}
 rupd();
</script>"""


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "run7")
