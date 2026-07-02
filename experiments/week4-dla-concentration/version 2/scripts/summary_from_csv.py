#!/usr/bin/env python3
"""Rebuild the combined R(t) + mass-radius summary figure from the saved
per-run CSVs (no video decode) -- used when the runs were processed in
separate invocations, so no single process held all three results."""

import csv
from pathlib import Path

import numpy as np

import enclosing_radius as er


def load(run):
    path = er.DATA / f"radius_{run['tag']}.csv"
    ppm = dpitch = None
    body = []
    for line in open(path):
        if line.startswith("#"):
            if "px_per_mm" in line:
                ppm, dpitch = (float(x) for x in line.split("=")[1].split("+/-"))
            continue
        body.append(line)
    r = list(csv.DictReader(body))
    g = lambda k: np.array([float(x[k]) for x in r])
    t, M, Rg, Rc, edge = g("t_s"), g("M_px"), g("Rg_px"), g("circ_R_px"), g("edge")
    res = dict(run=run, t=t, M=M, Rg=Rg, Rc=Rc, edge=edge,
               px_per_mm=ppm, dpitch=dpitch)
    res["t0"] = er.nucleation_time(t, M)
    res["win"] = er.growth_window(M, Rg, edge)
    tau = t - res["t0"]
    m = res["win"] & (tau > 0)
    if m.sum() >= 8:
        res["D"], res["dD"], _ = er.fit_loglog(Rg[m], M[m])
        res["beta"], res["dbeta"], _ = er.fit_loglog(tau[m], Rc[m])
    return res


def main():
    results = [load(run) for run in er.RUNS]
    out = er.summary_figure(results)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
