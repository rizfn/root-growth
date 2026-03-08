"""
viz_lc2_timeseries.py
=====================
For each IC = ±2 timeseries produced by runTimeseriesLC2.sh, plots the last
12τ of θ(t) and its derivative dθ/dt = −k·sin(θ(t−τ)), computed directly
from the recorded data using the τ-shifted signal.

Output: plots/lc2/timeseries/tauK_<value>_ic_<ic>.png
"""

import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Times where y changes sign, found by linear interpolation."""
    signs = np.sign(y)
    idx   = np.where(np.diff(signs) != 0)[0]
    if len(idx) == 0:
        return np.array([])
    # linear interpolation within each bracketing interval
    t0, t1 = t[idx], t[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    return t0 - y0 * (t1 - t0) / (y1 - y0)


def plot_timeseries(t, theta, tau, k, ic, plot_dir):
    record_dt    = float(t[1] - t[0])
    delay_steps  = round(tau / record_dt)
    window_steps = round(12 * tau / record_dt)

    # Need at least delay_steps extra points before the plot window
    needed = window_steps + delay_steps
    if len(t) < needed:
        print(f"  Not enough data (need {needed} pts, have {len(t)}) — skipping.")
        return

    # Extract the plot window (last 12τ) plus the preceding τ for delay lookup
    seg_t     = t[-(window_steps + delay_steps):]
    seg_theta = theta[-(window_steps + delay_steps):]

    # Derivative over the last 12τ only
    t_plot     = seg_t[delay_steps:]
    theta_plot = seg_theta[delay_steps:]
    theta_del  = seg_theta[:window_steps]        # θ(t − τ) for each plot point
    dtheta     = -k * np.sin(theta_del)

    crossings_d  = zero_crossings(t_plot, dtheta)                        # dθ/dt = 0
    crossings_pi = np.sort(np.concatenate([                               # θ = 0 or ±π
        zero_crossings(t_plot, theta_plot),
        zero_crossings(t_plot, np.abs(theta_plot) - np.pi),
    ]))

    tauk   = tau * k
    ic_str = f"{ic:+.0f}".replace("+", "p").replace("-", "n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"hspace": 0.15})

    ax1.plot(t_plot, theta_plot, color="#2166AC", lw=0.8)
    ax1.axhline( np.pi, color="gray", lw=0.8, ls="--", alpha=0.6, label=r"$\pm\pi$")
    ax1.axhline(-np.pi, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax1.set_ylabel(r"$\theta(t)$", fontsize=12)
    ax1.set_title(
        rf"$\tau k = {tauk:.4g}$,  IC = ${ic:+.0f}$  —  last $12\tau$ of timeseries",
        fontsize=12,
    )
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    ax2.plot(t_plot, dtheta, color="#D6604D", lw=0.8)
    ax2.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xlabel(r"$t$", fontsize=12)
    ax2.set_ylabel(r"$\dot\theta = -k\sin(\theta(t-\tau))$", fontsize=12)
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    # ── green: dθ/dt zero crossings — lines on both, labels on bottom ────────
    for ax in (ax1, ax2):
        for xc in crossings_d:
            ax.axvline(xc, color="green", lw=0.7, ls=":", alpha=0.7)

    y2_lo, y2_hi = ax2.get_ylim()
    label_y2 = y2_lo + 0.88 * (y2_hi - y2_lo)
    for i in range(len(crossings_d) - 1):
        xc0, xc1 = crossings_d[i], crossings_d[i + 1]
        ax2.text((xc0 + xc1) / 2, label_y2, rf"${(xc1 - xc0) / tau:.2f}\tau$",
                 ha="center", va="center", fontsize=7, color="green", clip_on=True)

    # ── purple: |θ| = π crossings — lines on both, labels on top ────────────
    for ax in (ax1, ax2):
        for xc in crossings_pi:
            ax.axvline(xc, color="purple", lw=0.7, ls=":", alpha=0.7)

    y1_lo, y1_hi = ax1.get_ylim()
    label_y1 = y1_lo + 0.88 * (y1_hi - y1_lo)
    for i in range(len(crossings_pi) - 1):
        xc0, xc1 = crossings_pi[i], crossings_pi[i + 1]
        ax1.text((xc0 + xc1) / 2, label_y1, rf"${(xc1 - xc0) / tau:.2f}\tau$",
                 ha="center", va="center", fontsize=7, color="purple", clip_on=True)

    os.makedirs(plot_dir, exist_ok=True)
    out = os.path.join(plot_dir, f"tauK_{tauk:.4g}_ic_{ic_str}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    TAU = 25
    K   = 0.164081
    IC  = 2.0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts_dir     = os.path.join(script_dir, "outputs", "timeseries")
    plot_dir   = os.path.join(script_dir, "plots", "lc2", "timeseries")

    ic_str = f"{IC:.6f}" if IC >= 0 else f"n{abs(IC):.6f}"
    pattern = os.path.join(ts_dir, f"tau_{TAU}_k_{K}_ic_{ic_str}_*.tsv")
    matches = glob.glob(pattern)
    if not matches:
        raise RuntimeError(f"No file found matching {pattern!r}. Run runTimeseriesLC2.sh first.")

    data = np.loadtxt(matches[0], skiprows=1)
    plot_timeseries(data[:, 0], data[:, 1], TAU, K, IC, plot_dir)


if __name__ == "__main__":
    main()
