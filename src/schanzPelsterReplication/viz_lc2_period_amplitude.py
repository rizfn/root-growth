"""
viz_lc2_period_amplitude.py
===========================
Reads second-limit-cycle timeseries (IC = ±2) produced by runTimeseriesLC2.sh,
computes amplitude and period for each τ·k value, and saves two figures:

  plots/lc2/amplitude_period_lc2.png   –  amplitude & period vs τ·k
  plots/lc2/period_correction_lc2.png  –  log-log (T – 4τ) vs ε = R – π
                                           with the theoretical prediction
                                           T – 4τ = 4√(8τ²ε/π³)
"""

import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def parse_filename(path: str) -> dict | None:
    name = os.path.basename(path).removesuffix(".tsv")
    try:
        decode = lambda s: -float(s[1:]) if s.startswith("n") else float(s)
        tau = float(name.split("tau_")[1].split("_k_")[0])
        k   = float(name.split("_k_")[1].split("_ic_")[0])
        ic  = decode(name.split("_ic_")[1].split("_twarmup_")[0])
        return dict(tau=tau, k=k, ic=ic)
    except (IndexError, ValueError):
        return None


def mean_amplitude(t: np.ndarray, theta: np.ndarray, tau: float, period: float | None = None) -> float:
    """Mean half-amplitude over non-overlapping windows.

    By default the window is `4*tau` (as appropriate for the first limit
    cycle).  For the second limit cycle the period can be longer; if
    ``period`` is supplied it will be used in place of ``4*tau`` (a safety
    minimum of ``4*tau`` is still enforced).
    """
    # choose a window length large enough to capture at least one cycle
    window = 4.0 * tau if period is None or np.isnan(period) else max(4.0 * tau, period)
    segs = [
        theta[(t >= t[0] + i * window) & (t < t[0] + (i + 1) * window)]
        for i in range(int((t[-1] - t[0]) / window))
    ]
    amps = [(s.max() - s.min()) / 2 for s in segs if len(s) > 1]
    return float(np.mean(amps)) if amps else float("nan")


def dominant_period(t: np.ndarray, theta: np.ndarray) -> float:
    """Period via first prominent peak of the normalised ACF, or nan."""
    dt = float(t[1] - t[0])
    n  = len(theta)
    theta_c = theta - np.mean(theta)
    fft_sig = np.fft.rfft(theta_c, n=2 * n)
    acf = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n].real
    if acf[0] == 0:
        return float("nan")
    acf /= acf[0]
    peaks, _ = find_peaks(acf, distance=max(1, int(5.0 / dt)), height=0.5)
    return float(peaks[0]) * dt if len(peaks) > 0 else float("nan")


def load_data(ts_dir: str, target_ics: set) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (tauk_arr, amp_arr, per_arr, tau) for the given ICs."""
    amp_acc: dict[float, list[float]] = {}
    per_acc: dict[float, list[float]] = {}
    tau = 25.0

    for path in sorted(glob.glob(os.path.join(ts_dir, "*.tsv"))):
        meta = parse_filename(path)
        if meta is None or round(meta["ic"], 4) not in target_ics:
            continue
        try:
            data = np.loadtxt(path, skiprows=1)
        except Exception:
            continue
        if data.ndim != 2 or data.shape[0] < 2:
            continue

        tau   = meta["tau"]
        t, theta = data[:, 0], data[:, 1]
        n_drop   = max(1, len(t) // 5)   # discard first 20 % as extra transient
        t, theta = t[n_drop:], theta[n_drop:]
        tauk     = round(tau * meta["k"], 8)

        # compute period first so the amplitude window can adapt
        per = dominant_period(t, theta)
        amp = mean_amplitude(t, theta, tau, period=per)
        if not np.isnan(amp):
            amp_acc.setdefault(tauk, []).append(amp)
        if not np.isnan(per):
            per_acc.setdefault(tauk, []).append(per)

    if not amp_acc:
        raise RuntimeError(f"No matching files in {ts_dir!r}. Run runTimeseriesLC2.sh first.")

    tauk_arr = np.array(sorted(amp_acc.keys()))
    amp_arr  = np.array([np.mean(amp_acc[tk]) for tk in tauk_arr])
    per_arr  = np.array([np.mean(per_acc[tk]) if tk in per_acc else np.nan for tk in tauk_arr])
    return tauk_arr, amp_arr, per_arr, tau


def plot_amplitude_period(tauk_arr, amp_arr, per_arr, tau, plot_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={"hspace": 0.38})

    ax1.scatter(tauk_arr, amp_arr, s=8, color="#2166AC", alpha=0.85, zorder=3, label=r"IC = $\pm 2$")
    ax1.axhline(np.pi, color="gray", lw=1.2, ls="--", alpha=0.7, label=r"$R = \pi$  (1st / 2nd LC boundary)")
    ax1.set_xlabel(r"$\tau k$", fontsize=12)
    ax1.set_ylabel(r"Average half-amplitude $R$", fontsize=12)
    ax1.set_title(r"Oscillation amplitude vs $\tau k$  (IC = $\pm 2$)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    ax2.scatter(tauk_arr, per_arr, s=8, color="#D6604D", alpha=0.85, zorder=3, label=r"IC = $\pm 2$")
    ax2.axhline(4 * tau, color="gray", lw=1.2, ls="--", alpha=0.7, label=rf"$4\tau = {4 * tau:.0f}$")
    ax2.set_xlabel(r"$\tau k$", fontsize=12)
    ax2.set_ylabel(r"Period $T$", fontsize=12)
    ax2.set_title(r"Oscillation period vs $\tau k$  (IC = $\pm 2$)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(r"Second limit cycle: amplitude and period  ($\tau = 25$)", fontsize=13)

    out = os.path.join(plot_dir, "amplitude_period_lc2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_period_correction(tauk_arr, amp_arr, per_arr, tau, plot_dir):
    mask  = (amp_arr > np.pi) & np.isfinite(per_arr) & (per_arr > 4 * tau)
    eps   = (amp_arr - np.pi)[mask]
    delta = (per_arr - 4 * tau)[mask] / tau    # in units of τ
    tauk  = tauk_arr[mask]
    valid = (eps > 0) & (delta > 0)
    eps, delta, tauk = eps[valid], delta[valid], tauk[valid]

    if len(eps) == 0:
        print("No 2nd-LC data with R > π and T > 4τ — skipping log-log plot.")
        return

    eps_th   = np.geomspace(eps.min() * 0.5, eps.max() * 2.0, 300)
    delta_th = 4.0 * np.sqrt(8.0 * tau**2 * eps_th / np.pi**3) / tau

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(eps, delta, c=tauk, cmap="viridis", s=14, alpha=0.85, zorder=3)
    # ax.plot(eps_th, delta_th, color="#D6604D", lw=2.0, ls="--", zorder=4,
    #         label=(r"theory:  $(T - 4\tau)/\tau = 4\sqrt{8\varepsilon/\pi^3}$"
    #                "\n" r"(slope $= 1/2$ on log-log)"))
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\tau k$", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon = R - \pi$", fontsize=13)
    ax.set_ylabel(r"$(T - 4\tau)\,/\,\tau$", fontsize=13)
    ax.set_title(r"Period correction vs amplitude excess above $\pi$"
                 "\n" r"Second limit cycle,  $\tau = 25$", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    out = os.path.join(plot_dir, "period_correction_lc2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir   = os.path.join(script_dir, "plots", "lc2")
    os.makedirs(plot_dir, exist_ok=True)

    tauk_arr, amp_arr, per_arr, tau = load_data(
        os.path.join(script_dir, "outputs", "timeseries"),
        target_ics={2.0, -2.0},
    )

    plot_amplitude_period(tauk_arr, amp_arr, per_arr, tau, plot_dir)
    plot_period_correction(tauk_arr, amp_arr, per_arr, tau, plot_dir)


if __name__ == "__main__":
    main()
