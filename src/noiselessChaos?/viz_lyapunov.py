import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.optimize import curve_fit


DELTAS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


def forward_smooth(sep, dt, tau):
    """
    Causal forward rolling mean over one oscillation period (4τ).

    Each value at index i is replaced by the mean of sep[i : i + window],
    where window = round(4τ / dt).  The final (window - 1) points are dropped,
    so the returned array is shorter than sep by (window - 1) elements.
    Implemented via cumsum for O(n) performance.
    """
    window = max(1, int(round(4 * tau / dt)))
    cs = np.cumsum(np.concatenate([[0.0], sep]))
    return (cs[window:] - cs[:-window]) / window   # length = len(sep) - window + 1


def log_saturation_model(t, C, lam, t_star):
    """
    Smooth ramp-then-plateau model in log-space:

        log y = C + lam*t          for t << t_star  (exponential growth)
        log y = C + lam*t_star     for t >> t_star  (plateau)

    The smooth interpolation uses a logistic transition:
        log y = C + lam*t_star - logaddexp(0, lam*(t_star - t))

    This is numerically stable via numpy.logaddexp.
    Free parameters: C (log of initial amplitude), lam (Lyapunov exponent),
    t_star (transition time from growth to saturation).
    """
    return C + lam * t_star - np.logaddexp(0.0, lam * (t_star - t))


def fit_saturation(t_s, env):
    """
    Fit log_saturation_model to the envelope.  Returns (lam, t_star, C, t_fit, env_fit)
    where t_fit/env_fit are the data used (above noise floor), or None on failure.
    """
    mask = env > 1e-13
    if mask.sum() < 20:
        return None

    t_fit   = t_s[mask]
    log_env = np.log(env[mask])

    # Initial guesses
    C0      = log_env[0]
    # Slope from first quarter of points
    q = max(1, len(t_fit) // 4)
    lam0    = max((log_env[q] - log_env[0]) / (t_fit[q] - t_fit[0] + 1e-9), 1e-4)
    t_star0 = t_fit[len(t_fit) // 2]

    try:
        popt, _ = curve_fit(
            log_saturation_model, t_fit, log_env,
            p0=[C0, lam0, t_star0],
            bounds=([-np.inf, -np.inf, t_fit[0]], [np.inf, np.inf, t_fit[-1]]),
            maxfev=10000,
        )
        return popt  # (C, lam, t_star)
    except (RuntimeError, ValueError):
        return None


def estimate_lyapunov(t, baseline, perturbed, tau):
    """
    Estimate the maximal Lyapunov exponent from a single perturbed trajectory.

    1. Compute absolute separation |perturbed - baseline|.
    2. Apply a causal forward rolling mean over 4τ, discarding the final 4τ
       window (where the average is not fully defined).  This averages out
       the oscillatory ripple and exposes the exponential envelope.
    3. Fit the smooth ramp-then-plateau model (log_saturation_model) to the
       full envelope above the noise floor.  The slope parameter is λ.

    Returns NaN if the fit fails or there are not enough valid points.
    """
    dt  = t[1] - t[0]
    sep = np.abs(perturbed - baseline)
    env = forward_smooth(sep, dt, tau)
    t_s = t[:len(env)]

    result = fit_saturation(t_s, env)
    if result is None:
        return np.nan
    _, lam, _ = result
    return lam


def load_and_estimate(filepath, tau):
    """
    Load a Lyapunov TSV and return (k, array of lambda estimates per delta).
    File columns: time, theta_baseline, theta_delta_X x N_DELTAS
    """
    data     = np.loadtxt(filepath, skiprows=1)
    t        = data[:, 0]
    baseline = data[:, 1]

    lambdas = []
    for p, delta in enumerate(DELTAS):
        perturbed = data[:, 2 + p]
        lam = estimate_lyapunov(t, baseline, perturbed, tau)
        lambdas.append(lam)

    # Parse k from filename
    # e.g. tau_25.0_k_0.210000_theta0_1.5708_twarmup_2000.0_tlyap_5000.0_dt_0.1.tsv
    basename = os.path.basename(filepath)
    k_str    = basename.split("_k_")[1].split("_theta0_")[0]
    return float(k_str), np.array(lambdas)


def main():
    tau      = 25
    theta0   = 1.5708
    t_lyap   = 5000
    dt       = 0.1    # record_dt

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir   = os.path.join(script_dir, "outputs", "lyapunov", "k_sweep")

    # Match all warmup phases for this tau/theta0/tlyap/dt combination
    pattern = f"tau_{tau}_k_*_theta0_{theta0}_twarmup_*_tlyap_{t_lyap}_dt_{dt}.tsv"
    all_files = sorted(glob.glob(os.path.join(base_dir, pattern)))

    if not all_files:
        print(f"No files found matching:\n  {os.path.join(base_dir, pattern)}")
        return

    # Group files by k value
    from collections import defaultdict
    files_by_k = defaultdict(list)
    for filepath in all_files:
        basename = os.path.basename(filepath)
        k_str = basename.split("_k_")[1].split("_theta0_")[0]
        files_by_k[float(k_str)].append(filepath)

    k_vals_sorted = sorted(files_by_k.keys())
    print(f"Found {len(k_vals_sorted)} k-values, "
          f"{len(all_files)} total files "
          f"({len(all_files)//len(k_vals_sorted)} phases each)")

    # Estimate lambda for every file; shape per k: (n_phases, N_DELTAS)
    lam_by_k = {}   # k -> array of shape (n_phases, N_DELTAS)
    for k, fps in files_by_k.items():
        phase_lams = []
        for fp in sorted(fps):
            _, lams = load_and_estimate(fp, tau)
            phase_lams.append(lams)
        lam_by_k[k] = np.array(phase_lams)  # (n_phases, N_DELTAS)

    k_vals  = np.array(k_vals_sorted)
    # Mean lambda: average over both phases and delta columns
    lam_mean = np.array([np.nanmean(lam_by_k[k]) for k in k_vals_sorted])
    lam_sem  = np.array([np.nanstd(lam_by_k[k]) / np.sqrt(np.sum(~np.isnan(lam_by_k[k])))
                         for k in k_vals_sorted])
    # Per-delta mean and SEM across phases
    lam_per_delta = np.array([np.nanmean(lam_by_k[k], axis=0) for k in k_vals_sorted])  # (n_k, N_DELTAS)
    lam_sem_per_delta = np.array([
        np.nanstd(lam_by_k[k], axis=0) / np.sqrt(np.sum(~np.isnan(lam_by_k[k]), axis=0).clip(1))
        for k in k_vals_sorted
    ])  # (n_k, N_DELTAS)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DELTAS)))

    # ── Plot 1: Lyapunov exponent vs k ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    for p, (delta, color) in enumerate(zip(DELTAS, colors)):
        lam_p = lam_per_delta[:, p]
        sem_p = lam_sem_per_delta[:, p]
        valid = ~np.isnan(lam_p)
        ax.errorbar(k_vals[valid], lam_p[valid], yerr=sem_p[valid],
                    fmt="o-", color=color, markersize=4, linewidth=1.2,
                    capsize=3, elinewidth=0.8, label=f"δ = {delta:.0e}")

    ax.errorbar(k_vals, lam_mean, yerr=lam_sem,
                fmt="k--", linewidth=1.5, capsize=3, elinewidth=0.8,
                label="mean", zorder=5)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("k (gravitropic strength)")
    ax.set_ylabel("Maximal Lyapunov exponent λ")
    ax.set_title(f"Lyapunov exponent vs k  (τ={tau}, θ₀={theta0})")
    # ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir = os.path.join(script_dir, "plots", "lyapunov")
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"lyapunov_vs_k_tau_{tau}.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")

    # ── Plot 2: separation traces for representative k (one phase each) ──────
    n_sample = min(6, len(k_vals_sorted))
    indices  = np.round(np.linspace(0, len(k_vals_sorted) - 1, n_sample)).astype(int)

    fig2, axes = plt.subplots(2, 3, figsize=(14, 8)) if n_sample == 6 else \
                 plt.subplots(1, n_sample, figsize=(5 * n_sample, 4))
    axes = np.array(axes).flatten()

    for ax_idx, k_idx in enumerate(indices):
        k_val = k_vals_sorted[k_idx]
        # Use the first phase file for the trace plot
        filepath = sorted(files_by_k[k_val])[0]
        data     = np.loadtxt(filepath, skiprows=1)
        t        = data[:, 0] - data[0, 0]   # relative time
        baseline = data[:, 1]
        rec_dt   = t[1] - t[0]

        ax = axes[ax_idx]
        for p, (delta, color) in enumerate(zip(DELTAS, colors)):
            perturbed = data[:, 2 + p]
            sep = np.abs(perturbed - baseline)
            env = forward_smooth(sep, rec_dt, tau)
            t_s = t[:len(env)]

            valid = env > 1e-15
            if valid.sum() < 5:
                continue

            # Forward-averaged envelope on log-lin axis
            ax.semilogy(t_s[valid], env[valid], color=color, linewidth=1.2,
                        label=f"δ={delta:.0e}" if ax_idx == 0 else None)

            # Fit the ramp-then-plateau model and overlay it
            result = fit_saturation(t_s, env)
            if result is not None:
                C_fit, lam_fit, t_star_fit = result
                t_overlay = t_s[env > 1e-15]
                y_overlay = np.exp(log_saturation_model(t_overlay, C_fit, lam_fit, t_star_fit))
                ax.semilogy(t_overlay, y_overlay,
                            "--", color=color, linewidth=1.2, alpha=0.85)

        n_phases = len(files_by_k[k_val])
        lam_str = f"λ≈{np.nanmean(lam_by_k[k_val]):.4f}"
        ax.set_title(f"k={k_val:.5f},  {lam_str}  ({n_phases} phases)", fontsize=9)
        ax.set_xlabel("time (post-warmup)")
        ax.set_ylabel("⟨|Δθ|⟩  (4τ forward avg)")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax_idx in range(n_sample, len(axes)):
        axes[ax_idx].set_visible(False)

    if n_sample == 6:
        handles, labels = axes[0].get_legend_handles_labels()
        fig2.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)
    fig2.tight_layout()

    outfile2 = os.path.join(output_dir, f"separation_traces_tau_{tau}.png")
    fig2.savefig(outfile2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {outfile2}")


if __name__ == "__main__":
    main()
