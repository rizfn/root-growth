import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.signal import find_peaks


def compute_period(data, min_acf_peak=0.5):
    """
    Estimate the oscillation period from the autocorrelation of the last
    quarter of the timeseries.

    Returns the period in time units, or NaN if the signal is aperiodic
    (no ACF peak above min_acf_peak), or 0 if the signal is constant.
    """
    time  = data[:, 0]
    theta = data[:, 1]

    # Use the last quarter to avoid transients
    start_idx = len(time) * 3 // 4
    time  = time[start_idx:]
    theta = theta[start_idx:]

    dt      = time[1] - time[0]
    n       = len(theta)
    theta_c = theta - np.mean(theta)

    # Fast ACF via FFT
    fft_sig = np.fft.rfft(theta_c, n=2 * n)
    acf     = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n]
    if acf[0] == 0:
        return 0.0  # constant signal

    acf /= acf[0]
    min_dist = max(1, int(0.03 * n))
    peaks, _ = find_peaks(acf, distance=min_dist, height=min_acf_peak)
    return peaks[0] * dt if len(peaks) > 0 else np.nan


def main():
    tau    = 20
    eta    = 0
    theta0 = 1.5708
    dt     = 0.1      # record_dt
    tmax   = 8000
    sim_no = 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir   = os.path.join(script_dir, "outputs", "SDDETimeseries", "k_sweep")

    pattern = f"tau_{tau}_k_*_theta0_{theta0}_dt_{dt}_tmax_{tmax}"
    folders = sorted(glob.glob(os.path.join(base_dir, pattern)))

    if not folders:
        print(f"No folders found matching:\n  {os.path.join(base_dir, pattern)}")
        return

    k_vals   = []
    periods  = []
    periodic = []

    for folder in folders:
        basename = os.path.basename(folder)
        k_str = basename.split("_k_")[1].split("_theta0_")[0]
        k_val = float(k_str)

        tsv_pattern = os.path.join(folder, f"eta_{eta}_simNo_{sim_no}.tsv")
        files = glob.glob(tsv_pattern)
        if not files:
            print(f"  No file found: {tsv_pattern}")
            continue

        data = np.loadtxt(files[0], skiprows=1)
        T = compute_period(data)

        k_vals.append(k_val)
        periods.append(T)
        periodic.append(not np.isnan(T) and T > 0)

    k_vals  = np.array(k_vals)
    periods = np.array(periods)
    periodic = np.array(periodic)

    print(f"Found {len(k_vals)} k-values: "
          f"{periodic.sum()} periodic, {(~periodic).sum()} aperiodic/constant")

    # Normalise period by tau
    periods_norm = periods / tau

    # ── Plot: period / τ vs k ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_vals[periodic], periods_norm[periodic],
            "o-", color="steelblue", markersize=5, linewidth=1.4)
    ax.scatter(k_vals[~periodic], np.full(np.sum(~periodic), np.nan),
               marker="x", color="crimson", s=40, zorder=5, label="aperiodic")
    ax.axhline(4, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("k (gravitropic strength)")
    ax.set_ylabel("Period / τ")
    ax.set_title(f"Oscillation period vs k  (τ={tau}, η={eta}, θ₀={theta0})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    output_dir = os.path.join(script_dir, "plots", "k_sweep")
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"period_vs_k_tau_{tau}.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
