"""
Ensemble MSD: average time-averaged MSDs over multiple trajectories.

Usage
-----
    from scripts.analysis.ensemble_msd import ensemble_msd
    tau, msd_mean, msd_sem = ensemble_msd(trajectories, fps)

where `trajectories` is a list of (N_i, 2) arrays (positions in µm)
and `fps` is the camera frame rate in Hz.
"""

import numpy as np
from .msd import compute_msd_single


def ensemble_msd(
    trajectories: list[np.ndarray],
    fps: float,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    tau      : lag times in seconds
    msd_mean : ensemble-mean MSD in µm²
    msd_sem  : standard error on the mean across trajectories
    """
    if max_lag is None:
        min_len = min(len(t) for t in trajectories)
        max_lag = min_len // 4

    msds = []
    for traj in trajectories:
        lags, msd = compute_msd_single(traj, max_lag=max_lag)
        msds.append(msd)

    msds = np.array(msds)           # (n_tracks, max_lag)
    msd_mean = np.mean(msds, axis=0)
    msd_sem  = np.std(msds, axis=0, ddof=1) / np.sqrt(len(trajectories))
    tau = lags / fps
    return tau, msd_mean, msd_sem
