"""
viz_intermittency.py
====================
Intermittency analysis for the deterministic DDE:
    dθ/dt = -k·sin(θ(t-τ))

Plots
-----
1. Time series with laminar phases highlighted
2. Stroboscopic (Poincaré) return map at Δt = 4τ
3. Return map minus diagonal
4. Laminar-length distribution — log-log (power-law) vs log-linear (exponential)

Expected signatures for different intermittency types:
  Type I (tangent):     return map channel above diagonal, P(ℓ) ~ ℓ^(-3/2)
  Type III (subcrit.):  similar P(ℓ) scaling
  Crisis-induced:       exponential P(ℓ), iterates hitting unstable orbit
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── Parameters (must match runIntermittency.sh) ───────────────────────────────
TAU       = 25
THETA0    = 1.5708
T_WARMUP  = 5000
# T_MEASURE = 1000_000
T_MEASURE = '1e+06'
DT_REC    = 0.1        # record_dt

# Stroboscopic interval: 4τ is the natural period of the DDE
STROBE_DT = 4 * TAU

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_file(filepath):
    """Load time series TSV → (t, theta) arrays."""
    data  = np.loadtxt(filepath, skiprows=1)
    return data[:, 0], data[:, 1]


def stroboscopic(t, theta, strobe_dt):
    """
    Sample θ at every multiple of strobe_dt (starting from t[0] rounded up).
    Returns array of θ values at those times via nearest-neighbour lookup.
    Used only for the return-map plots (Test 2).
    """
    t0      = t[0]
    n_max   = int((t[-1] - t0) / strobe_dt)
    targets = t0 + np.arange(1, n_max + 1) * strobe_dt
    indices = np.searchsorted(t, targets)
    indices = np.clip(indices, 0, len(t) - 1)
    return theta[indices]


def detect_laminar_phases(theta, tau, record_dt, jump_threshold=np.pi, min_periods=1):
    """
    Detect laminar phases based on stability of the oscillation center.

    During laminar phases θ oscillates around a roughly constant center
    (e.g. 0→2π with center ≈ π).  A chaotic burst shifts the center by
    ~2π (e.g. to 2π→4π, center ≈ 3π).

    Strategy
    --------
    1. Compute running mean of θ over one period (4τ) → oscillation center(t)
    2. Compare center(t) to center(t − 4τ): small change → laminar
    3. Discard laminar runs shorter than min_periods × 4τ

    Parameters
    ----------
    theta          : array of θ values
    tau            : delay parameter
    record_dt      : recording time step
    jump_threshold : maximum |Δcenter| to be considered laminar (default π)
    min_periods    : minimum laminar duration in units of 4τ (default 1)

    Returns
    -------
    lengths_t : 1-D array of laminar-phase durations (time units)
    is_lam    : bool array (length = len(theta) − n_period)
    """
    n_period    = int(round(4 * tau / record_dt))
    min_samples = int(round(min_periods * n_period))

    if 2 * n_period >= len(theta):
        return np.array([]), np.zeros(max(0, len(theta) - n_period), dtype=bool)

    # Rolling mean over one full period → oscillation center
    center = uniform_filter1d(theta, size=n_period, mode='nearest')

    # Drift of center over one period
    delta_center = np.abs(center[n_period:] - center[:-n_period])
    is_lam = delta_center < jump_threshold

    # Remove laminar runs shorter than min_periods × 4τ
    filtered = np.copy(is_lam)
    lengths_t = []
    i = 0
    while i < len(filtered):
        if filtered[i]:
            start = i
            while i < len(filtered) and filtered[i]:
                i += 1
            run_len = i - start
            if run_len < min_samples:
                filtered[start:i] = False
            else:
                lengths_t.append(run_len * record_dt)
        else:
            i += 1

    return np.array(lengths_t), filtered


def fit_powerlaw(lengths, ell_min=5):
    """
    Maximum-likelihood estimate of the power-law exponent α given the
    integer samples ℓ ≥ ell_min (Clauset et al. 2009 discrete MLE).
    Returns α estimate and its standard error.
    """
    x = lengths[lengths >= ell_min].astype(float)
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    alpha = 1.0 + n / np.sum(np.log(x / (ell_min - 0.5)))
    se    = (alpha - 1.0) / np.sqrt(n)
    return alpha, se


def fit_exponential(lengths, ell_min=5):
    """MLE rate λ for exponential distribution P(ℓ) = λ·exp(-λ·ℓ)."""
    x = lengths[lengths >= ell_min].astype(float)
    if len(x) < 10:
        return np.nan, np.nan
    lam = 1.0 / np.mean(x)
    se  = lam / np.sqrt(len(x))
    return lam, se


def ks_powerlaw_vs_exp(lengths, ell_min=5):
    """
    Generate synthetic samples from best-fit power-law and exponential;
    compare each to data via KS distance.
    Lower KS = better fit.
    """
    x = lengths[lengths >= ell_min].astype(float)
    if len(x) < 20:
        return np.nan, np.nan

    alpha_ml, _ = fit_powerlaw(lengths, ell_min)
    lam_ml,   _ = fit_exponential(lengths, ell_min)

    # Empirical CDF
    xs   = np.sort(x)
    n    = len(xs)
    ecdf = np.arange(1, n + 1) / n

    # Power-law CDF: P(>ℓ) = (ℓ/ℓ_min)^{-(α-1)}
    pl_cdf  = 1.0 - (xs / ell_min)**(-(alpha_ml - 1))
    pl_cdf  = np.clip(pl_cdf, 0, 1)

    # Exponential CDF: 1 - exp(-λ·ℓ)
    exp_cdf = 1.0 - np.exp(-lam_ml * xs)

    ks_pl  = np.max(np.abs(ecdf - pl_cdf))
    ks_exp = np.max(np.abs(ecdf - exp_cdf))
    return ks_pl, ks_exp


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir   = os.path.join(script_dir, "outputs", "intermittency", "k_sweep")
    output_dir = os.path.join(script_dir, "plots", "intermittency")
    os.makedirs(output_dir, exist_ok=True)

    pattern   = (f"tau_{TAU}_k_*_theta0_{THETA0}_twarmup_{T_WARMUP}"
                 f"_tmeasure_{T_MEASURE}_dt_{DT_REC}.tsv")
    all_files = sorted(glob.glob(os.path.join(base_dir, pattern)))

    if not all_files:
        print(f"No files found matching:\n  {os.path.join(base_dir, pattern)}")
        return

    # Parse k from filename
    def parse_k(fp):
        return float(os.path.basename(fp).split("_k_")[1].split("_theta0_")[0])

    k_file = {parse_k(fp): fp for fp in all_files}
    k_vals = np.array(sorted(k_file.keys()))
    print(f"Found {len(k_vals)} k-values in [{k_vals.min():.4f}, {k_vals.max():.4f}]")

    # Pre-compute laminar intervals for every k.
    n_period = int(round(4 * TAU / DT_REC))      # 1000 samples = one oscillation
    print(f"n_period = {n_period} samples  (4τ = {4*TAU} time units at dt={DT_REC})")
    print("Pre-computing laminar intervals for all k values...")

    lam_lengths = {}   # k -> array of laminar durations in time units
    for k in k_vals:
        t, theta        = load_file(k_file[k])
        lengths_t, _    = detect_laminar_phases(theta, TAU, DT_REC)
        lam_lengths[k]  = lengths_t
        if len(lengths_t) > 0:
            print(f"  k={k:.5f}: {len(lengths_t)} intervals, "
                  f"⟨ℓ⟩={np.mean(lengths_t):.1f} t-units")
        else:
            print(f"  k={k:.5f}: 0 intervals")

    # ── Estimate k_c and choose reference k-values ────────────────────────────
    mean_lengths = np.array([np.mean(lam_lengths[k]) if len(lam_lengths[k]) >= 5
                             else np.nan for k in k_vals])
    std_lengths  = np.array([np.std(lam_lengths[k]) / np.sqrt(len(lam_lengths[k]))
                             if len(lam_lengths[k]) >= 5 else np.nan for k in k_vals])
    valid_mask   = ~np.isnan(mean_lengths) & (mean_lengths > 0)
    k_c_idx      = np.where(valid_mask)[0][np.argmax(mean_lengths[valid_mask])]
    k_c_est      = k_vals[k_c_idx]
    print(f"\n  Estimated k_c (peak ⟨ℓ⟩) ≈ {k_c_est:.5f}  (index {k_c_idx})")

    # Reference k's: just before k_c, at k_c, just after, well into chaos
    ref_idxs = np.unique(np.clip(
        [k_c_idx - 1, k_c_idx, k_c_idx + 1, min(k_c_idx + 4, len(k_vals) - 1)],
        0, len(k_vals) - 1))
    ref_k_vals = k_vals[ref_idxs]
    print(f"  Reference k-values: {[f'{k:.5f}' for k in ref_k_vals]}")

    # ── 1. Time series with laminar phases ────────────────────────────────────
    n_ts   = min(6, len(k_vals))
    idxs   = np.round(np.linspace(0, len(k_vals) - 1, n_ts)).astype(int)
    ncols  = 3
    nrows  = (n_ts + ncols - 1) // ncols

    fig1, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    axes_flat  = axes.flatten()

    for i, ki in enumerate(idxs):
        k        = k_vals[ki]
        t, theta = load_file(k_file[k])
        t_rel    = t - t[0]
        _, is_lam = detect_laminar_phases(theta, TAU, DT_REC)
        t_lam     = t_rel[n_period:]  # is_lam[j] aligns with t_rel[j + n_period]

        ax = axes_flat[i]
        mask_short = t_rel <= 5000
        ax.plot(t_rel[mask_short], theta[mask_short], lw=0.5, color="steelblue")

        # Shade laminar intervals
        short_mask = t_lam <= 5000
        is_lam_s   = is_lam[short_mask]
        t_lam_s    = t_lam[short_mask]
        if len(is_lam_s) > 1:
            transitions = np.where(np.diff(is_lam_s.astype(np.int8)))[0]
            edge_idx    = np.concatenate([[0], transitions + 1, [len(is_lam_s)]])
            for a, b in zip(edge_idx[:-1], edge_idx[1:]):
                if is_lam_s[a]:
                    ax.axvspan(t_lam_s[a], t_lam_s[min(b, len(t_lam_s) - 1)],
                               alpha=0.15, color="orange")

        lengths = lam_lengths[k]
        mean_l  = np.mean(lengths) if len(lengths) > 0 else 0
        ax.set_title(f"k = {k:.5f},  ⟨ℓ⟩ = {mean_l:.1f} t-units  "
                     f"(n={len(lengths)})", fontsize=9)
        ax.set_xlabel("time (post-warmup)")
        ax.set_ylabel("θ(t)")
        ax.grid(True, alpha=0.3)

    for j in range(n_ts, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig1.suptitle("Time series (orange = laminar phases, center-stability detector)",
                  fontsize=11)
    fig1.tight_layout()
    out1 = os.path.join(output_dir, "timeseries.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out1}")

    # ── 2. Stroboscopic return map ──────────────────────────────────────────────
    n_rm = len(ref_idxs)

    fig2, axes2 = plt.subplots(1, n_rm, figsize=(5 * n_rm, 5), squeeze=False)
    axes2 = axes2.flatten()

    for i, ki in enumerate(ref_idxs):
        k   = k_vals[ki]
        t, theta = load_file(k_file[k])
        theta_n  = stroboscopic(t, theta, STROBE_DT)

        ax = axes2[i]

        # Full return map
        ax.plot(theta_n[:-1], theta_n[1:], ".", markersize=1.5,
                color="steelblue", alpha=0.5)

        # Zoom channel: focus on where map is closest to diagonal
        theta_mid  = np.median(theta_n)
        half_range = np.std(theta_n) * 1.5
        lo, hi     = theta_mid - half_range, theta_mid + half_range
        diag       = np.linspace(lo, hi, 200)
        ax.plot(diag, diag, "k--", lw=1, zorder=5, label="y = x")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"k = {k:.5f}", fontsize=9)
        ax.set_xlabel("θ_n  (stroboscopic)")
        ax.set_ylabel("θ_{n+1}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig2.suptitle(f"Stroboscopic return map (Δt = 4τ = {STROBE_DT:.0f})\n"
                  "Type-I signature: narrow channel gap above diagonal", fontsize=10)
    fig2.tight_layout()
    out2 = os.path.join(output_dir, "return_map.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ── 3. Map minus diagonal ───────────────────────────────────────────────
    fig2b, axes2b = plt.subplots(1, n_rm, figsize=(5 * n_rm, 4), squeeze=False)
    axes2b = axes2b.flatten()
    for i, ki in enumerate(ref_idxs):
        k   = k_vals[ki]
        t, theta = load_file(k_file[k])
        theta_n = stroboscopic(t, theta, STROBE_DT)
        x_n, x_n1 = theta_n[:-1], theta_n[1:]
        axes2b[i].plot(x_n, x_n1 - x_n, ".", markersize=1.5,
                       color="darkorange", alpha=0.5)
        axes2b[i].axhline(0, color="k", lw=1, ls="--")
        axes2b[i].set_title(f"k = {k:.5f}", fontsize=9)
        axes2b[i].set_xlabel("θ_n")
        axes2b[i].set_ylabel("θ_{n+1} − θ_n")
        axes2b[i].grid(True, alpha=0.3)
    fig2b.suptitle("Map minus diagonal  (Type-I: parabola tangent to 0 at k_c)", fontsize=10)
    fig2b.tight_layout()
    out2b = os.path.join(output_dir, "return_map_minus_diagonal.png")
    fig2b.savefig(out2b, dpi=150, bbox_inches="tight")
    plt.close(fig2b)
    print(f"Saved: {out2b}")

    # ── 4. Laminar-length distribution ────────────────────────────────────────
    # Use the same k-values as the return maps
    dist_k_vals = [k_vals[ki] for ki in ref_idxs]
    n_dist      = len(dist_k_vals)
    ncols_d     = min(4, n_dist)
    nrows_d     = (n_dist + ncols_d - 1) // ncols_d

    fig_d, axes_d = plt.subplots(nrows_d, ncols_d, figsize=(6 * ncols_d, 5 * nrows_d),
                                 squeeze=False)
    axes_d_flat = axes_d.flatten()

    def plot_distribution(ax, lengths, title_prefix):
        if len(lengths) < 20:
            ax.text(0.5, 0.5, "Too few intervals", transform=ax.transAxes,
                    ha="center", va="center")
            return

        ell_min = 4 * TAU   # minimum laminar duration = one period
        alpha_ml, alpha_se = fit_powerlaw(lengths, ell_min)
        ks_pl, _           = ks_powerlaw_vs_exp(lengths, ell_min)

        # Logarithmic bins via geomspace; divide by bin widths for density
        lo_bin = max(4 * TAU, lengths.min())
        hi_bin = lengths.max() * 1.01
        bins    = np.geomspace(lo_bin, hi_bin, 40)
        counts, edges = np.histogram(lengths, bins=bins)
        widths  = np.diff(edges)
        total   = counts.sum()
        density = np.where(counts > 0, counts / (total * widths), 0)
        mids    = np.sqrt(edges[:-1] * edges[1:])   # geometric midpoints
        pos     = density > 0

        ax.loglog(mids[pos], density[pos], "o", color="steelblue",
                  markersize=4, label=f"data (n={len(lengths)})")
        if not np.isnan(alpha_ml):
            ll_fit = (alpha_ml - 1) * mids[pos][0]**(alpha_ml - 1) * mids[pos]**(-alpha_ml)
            scale  = density[pos][0] / ll_fit[0]
            ax.loglog(mids[pos], scale * ll_fit, "--", color="firebrick",
                      label=f"power-law α={alpha_ml:.2f}±{alpha_se:.2f}  KS={ks_pl:.3f}")
        # Reference slope -3/2
        ref_x = np.geomspace(mids[pos][0], mids[pos][-1], 100)
        ax.loglog(ref_x, density[pos][0] * (mids[pos][0] / ref_x)**1.5,
                  ":", color="gray", lw=1.2, label="slope −3/2 (Type I)")
        ax.set_xlabel("Laminar duration  ℓ  (time units)")
        ax.set_ylabel("P(ℓ)")
        ax.set_title(title_prefix, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

        print(f"  {title_prefix}: n={len(lengths)}, α={alpha_ml:.3f}±{alpha_se:.3f}, "
              f"KS_pl={ks_pl:.3f}")

    for i, k in enumerate(dist_k_vals):
        plot_distribution(axes_d_flat[i], lam_lengths[k], f"k={k:.5f}")

    for j in range(n_dist, len(axes_d_flat)):
        axes_d_flat[j].set_visible(False)

    fig_d.suptitle("Laminar-length distribution (log-log)\n"
                   "Power-law α≈3/2 → Type I/III", fontsize=10)
    fig_d.tight_layout()
    out_d = os.path.join(output_dir, "laminar_distribution.png")
    fig_d.savefig(out_d, dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"Saved: {out_d}")

    # ── 5. Mean laminar length scaling: ⟨ℓ⟩ vs ε = k − k_c ─────────────────
    # Plot only k > k_c (the chaotic side where ε > 0)
    above_kc = valid_mask & (k_vals > k_c_est)
    eps_arr  = k_vals[above_kc] - k_c_est
    ml_arr   = mean_lengths[above_kc]
    err_arr  = std_lengths[above_kc]

    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    if len(eps_arr) >= 2:
        ax_s.errorbar(eps_arr, ml_arr, yerr=err_arr,
                      fmt="o", color="steelblue", markersize=5,
                      capsize=3, elinewidth=0.8, label="data")

        # Fit log-log slope → β
        log_eps = np.log(eps_arr)
        log_ml  = np.log(ml_arr)
        coeffs  = np.polyfit(log_eps, log_ml, 1)
        beta_fit = -coeffs[0]
        eps_fit  = np.geomspace(eps_arr.min(), eps_arr.max(), 200)
        A_fit    = np.exp(coeffs[1])
        ax_s.loglog(eps_fit, A_fit * eps_fit**(-beta_fit), "--", color="firebrick",
                    label=f"fit: β = {beta_fit:.2f}")

        ylim = ax_s.get_ylim()

        # Reference slopes
        for b_ref, ls_ref, lbl in [(0.5, ":", "β=0.5 (Type I/III)"),
                                    (1.0, "-.", "β=1.0 (crisis)")]:
            A_ref = ml_arr[0] * eps_arr[0]**b_ref
            ax_s.loglog(eps_fit, A_ref * eps_fit**(-b_ref),
                        color="gray", ls=ls_ref, lw=1, alpha=0.6, label=lbl)

        ax_s.set_ylim(ylim)
        ax_s.legend(fontsize=9)

    ax_s.set_xlabel("ε = k − k_c")
    ax_s.set_ylabel("⟨ℓ⟩  (mean laminar duration, time units)")
    ax_s.set_title(f"Mean laminar length scaling  (k_c ≈ {k_c_est:.4f})")
    ax_s.grid(True, alpha=0.3, which="both")
    fig_s.tight_layout()
    out_s = os.path.join(output_dir, "mean_laminar_scaling.png")
    fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
    plt.close(fig_s)
    print(f"Saved: {out_s}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n═══ Intermittency classification summary ═══")
    print("  Check laminar_distribution.png:")
    print("    Power-law tail (log-log linear) → Type I/III")
    print("  Check return_map.png:")
    print("    Channel just above diagonal     → Type I (tangent bifurcation)")
    print("    Iterates hitting unstable orbit  → Crisis")
    print("  Check mean_laminar_scaling.png:")
    print("    Divergence of ⟨ℓ⟩ near k_c")


if __name__ == "__main__":
    main()
