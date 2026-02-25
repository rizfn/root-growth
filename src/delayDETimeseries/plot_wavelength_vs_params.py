import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from glob import glob
from scipy.signal import find_peaks


def compute_wavelength(data, min_acf_peak=0.5):
    time  = data[:, 0]
    theta = data[:, 1]

    start_idx = len(time) * 3 // 4
    time  = time[start_idx:]
    theta = theta[start_idx:]

    dt      = time[1] - time[0]
    n       = len(theta)
    theta_c = theta - np.mean(theta)

    fft_sig = np.fft.rfft(theta_c, n=2 * n)
    acf     = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n]
    if acf[0] == 0:
        return 0  # constant signal — wavelength is effectively 0

    acf /= acf[0]
    min_dist = max(1, int(0.03 * n))
    peaks, _ = find_peaks(acf, distance=min_dist, height=min_acf_peak)
    return peaks[0] * dt if len(peaks) > 0 else np.nan


def main():
    script_dir = os.path.dirname(__file__)
    raster_dir = os.path.join(script_dir, "outputs", "SDDETimeseries", "tau_k_raster")

    results   = {}
    aperiodic = set()

    for folder in glob(os.path.join(raster_dir, "tau_*")):
        if not os.path.isdir(folder):
            continue
        name = os.path.basename(folder)
        m = re.match(r"tau_([\d.]+)_k_([\d.]+)_theta0_[\d.]+_dt_([\d.]+)_tmax_[\d.]+", name)
        if not m:
            continue

        tau, k, dt = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if dt != 0.1:
            continue

        candidates = sorted(f for f in os.listdir(folder) if re.match(r"eta_0_simNo_\d+\.tsv", f))
        if not candidates:
            print(f"No eta_0 file in {name}")
            continue
        data = np.loadtxt(os.path.join(folder, candidates[0]), skiprows=1)

        wl = compute_wavelength(data)
        if np.isnan(wl):
            aperiodic.add((tau, k))
        else:
            results[(tau, k)] = wl / tau

    print(f"{len(results)} periodic, {len(aperiodic)} aperiodic/blown-up.")

    all_pairs = set(results.keys()) | aperiodic
    taus = sorted({t for t, _ in all_pairs})
    ks   = sorted({k for _, k in all_pairs})

    grid           = np.full((len(ks), len(taus)), np.nan)
    aperiodic_grid = np.zeros((len(ks), len(taus)), dtype=bool)

    for i, k in enumerate(ks):
        for j, tau in enumerate(taus):
            if (tau, k) in results:
                grid[i, j] = results[(tau, k)]
            elif (tau, k) in aperiodic:
                aperiodic_grid[i, j] = True

    output_dir = os.path.join(script_dir, "plots", "wavelength_analysis")
    os.makedirs(output_dir, exist_ok=True)

    tau_arr, k_arr = np.array(taus), np.array(ks)

    def log_edges(arr):
        la = np.log(arr)
        d  = np.diff(la)
        return np.exp(np.r_[la[0] - d[0]/2, la[:-1] + d/2, la[-1] + d[-1]/2])

    tau_edges = log_edges(tau_arr)
    k_edges   = log_edges(k_arr)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="lightgray")

    pcm = ax.pcolormesh(tau_edges, k_edges, np.ma.masked_invalid(grid),
                        cmap=cmap, vmin=0, vmax=8)

    for i, k in enumerate(ks):
        for j, tau in enumerate(taus):
            if aperiodic_grid[i, j]:
                ax.add_patch(plt.Rectangle(
                    (tau_edges[j], k_edges[i]),
                    tau_edges[j+1] - tau_edges[j],
                    k_edges[i+1]   - k_edges[i],
                    facecolor="#888888", edgecolor="none", alpha=0.75, zorder=2,
                ))

    cbar = fig.colorbar(pcm, ax=ax, label="Wavelength / τ")
    cbar.ax.axhline(4, color="black", linewidth=1.5, linestyle="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tau_edges[0], tau_edges[-1])
    ax.set_ylim(k_edges[0],   k_edges[-1])
    ax.set_xticks(tau_arr)
    ax.set_yticks(k_arr)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel("τ")
    ax.set_ylabel("k")
    ax.set_title("Normalised wavelength (λ / τ)  —  η = 0 raster scan")

    tau_line = np.geomspace(min(taus), max(taus), 500)
    line1, = ax.plot(tau_line, (np.pi / 2) / tau_line,
                     color="royalblue", linewidth=1.5, linestyle="--", label=r"$\tau k = \pi/2$")
    line2, = ax.plot(tau_line, 4 / tau_line,
                     color="green", linewidth=1.5, linestyle="--", label=r"$\tau k = 4$")
    line3, = ax.plot(tau_line, 5 / tau_line,
                     color="magenta", linewidth=1.5, linestyle="--", label=r"$\tau k = 5$")
    aperiodic_patch = mpatches.Patch(facecolor="#888888", alpha=0.75, label="Aperiodic / blown-up")
    ax.legend(handles=[line1, line2, line3, aperiodic_patch])

    plt.tight_layout()
    output_file = os.path.join(output_dir, "wavelength_heatmap.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    main()
