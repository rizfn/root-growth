import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def load_timeseries(folder):
    """Load a single .tsv timeseries from *folder* (first file found)."""
    files = glob.glob(os.path.join(folder, "*.tsv"))
    if not files:
        return None
    data = np.loadtxt(files[0], skiprows=1)
    return data


def main():
    tau = 25
    eta = 0.0
    theta0 = 1.5708
    dt = 0.1          # record_dt
    tmax = 4000
    sim_no = 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "outputs", "SDDETimeseries", "k_sweep")

    # Discover all k-sweep folders that match this tau
    pattern = f"tau_{tau}_k_*_theta0_{theta0}_dt_{dt}_tmax_{tmax}"
    folders = sorted(glob.glob(os.path.join(base_dir, pattern)))

    if not folders:
        print(f"No folders found matching {os.path.join(base_dir, pattern)}")
        return

    # Parse k values from folder names
    runs = []
    for folder in folders:
        basename = os.path.basename(folder)
        # e.g. tau_25.0_k_0.3_theta0_1.5708_dt_0.1_tmax_4000.0
        parts = basename.split("_k_")[1]
        k_str = parts.split("_theta0_")[0]
        k_val = float(k_str)
        data = load_timeseries(folder)
        if data is not None:
            runs.append((k_val, data))

    runs.sort(key=lambda x: x[0])
    print(f"Found {len(runs)} k-values for tau={tau}")

    # ── Grid of individual timeseries ────────────────────────────
    n = len(runs)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             sharex=True, sharey=False)
    axes = axes.flatten()

    for idx, (k_val, data) in enumerate(runs):
        ax = axes[idx]
        t = data[:, 0]
        theta = data[:, 1]
        # Show last quarter to see steady-state behaviour
        mask = t >= 3 * tmax / 4
        ax.plot(t[mask], theta[mask], linewidth=0.5, alpha=0.8)
        ax.set_title(f"k={k_val:.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel("Time")
    fig.supylabel("θ (radians)")
    fig.suptitle(f"Period-doubling sweep: τ={tau}, η={eta}, k ∈ [{runs[0][0]:.3f}, {runs[-1][0]:.3f}]", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_dir = os.path.join(script_dir, "plots", "k_sweep")
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"k_sweep_grid_tau_{tau}.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {outfile}")

if __name__ == "__main__":
    main()
