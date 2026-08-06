import numpy as np


def read_crossings(path):
    """Return list of arrays of upward-zero-crossing times, one per realisation."""
    out = []
    with open(path, "rb") as f:
        nreal = np.fromfile(f, dtype=np.int32, count=1)[0]
        for _ in range(nreal):
            nc = np.fromfile(f, dtype=np.int64, count=1)[0]
            out.append(np.fromfile(f, dtype=np.float64, count=nc))
    return out


def phase_diffusion(path, nskip=5, return_curve=False):
    """D from ensemble variance of the n-th crossing time.

    Var_ens(t_n) = 2 D t_n / omega^2,  omega = 2 pi / T.
    Returns (D, T, ncross_used).
    """
    xs = read_crossings(path)
    nmin = min(len(x) for x in xs)
    if nmin < 20:
        return np.nan, np.nan, nmin
    M = np.vstack([x[:nmin] for x in xs])          # (nreal, nmin)
    n = np.arange(nmin)
    tbar = M.mean(axis=0)
    var = M.var(axis=0, ddof=1)
    # mean period from a linear fit of tbar vs n
    T = np.polyfit(n[nskip:], tbar[nskip:], 1)[0]
    omega = 2 * np.pi / T
    # slope of Var vs mean crossing time, forced through the first usable point
    x = tbar[nskip:] - tbar[nskip]
    y = var[nskip:] - var[nskip]
    slope = np.sum(x * y) / np.sum(x * x)          # least squares, no intercept
    D = slope * omega**2 / 2.0
    if return_curve:
        return D, T, nmin, tbar, var
    return D, T, nmin
