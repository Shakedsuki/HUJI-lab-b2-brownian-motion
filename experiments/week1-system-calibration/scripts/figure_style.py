"""
figure_style.py  (week1-system-calibration)
-------------------------------------------
Consistent matplotlib styling so every figure in the report matches. Call
set_style() at the top of a plotting script; savefig() writes into this week's
figures/ dir.
"""

import os
import matplotlib.pyplot as plt

import _paths   # sibling module (same scripts/ dir is on sys.path at runtime)


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "figure.figsize": (6.0, 4.2),
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(name, fig=None, **kwargs):
    """Save into this week's figures/ dir and return the path."""
    os.makedirs(_paths.FIGURES_DIR, exist_ok=True)
    path = os.path.join(_paths.FIGURES_DIR, name)
    (fig or plt).savefig(path, bbox_inches="tight", **kwargs)
    return path
