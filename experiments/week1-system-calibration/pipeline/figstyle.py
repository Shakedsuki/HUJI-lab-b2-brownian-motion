"""
figstyle.py  (pipeline)
-----------------------
Self-contained matplotlib styling for the v2 pipeline's figures (no dependency
on the old scripts/figure_style.py). set_style() once per plotting entry point;
save(fig, path) writes (creating parent dirs).
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
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


def save(fig, path, **kwargs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)
    return path
