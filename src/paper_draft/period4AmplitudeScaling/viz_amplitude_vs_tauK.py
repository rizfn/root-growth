import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import j1
from matplotlib.ticker import FuncFormatter

TARGET_ICS = {1.0, -1.0}


def hb_nharmonic(ktau_grid: np.ndarray, n_harm: int) -> tuple[np.ndarray, np.ndarray]:
    """
    N-harmonic balance for θ(t) = Σ_{m=1}^{n_harm} R_{2m-1} cos((2m-1)ωt), ω·τ = π/2.

    With ω·τ = π/2 the m-th harmonic acquires phase (2m-1)π/2, so:
        θ(t-τ) = Σ_m (-1)^{m+1} R_{2m-1} sin((2m-1)ωt)

    Projecting onto sin((2m-1)ωt) gives n_harm coupled equations solved numerically.
    n_harm=1 recovers the analytic J1 equation exactly.

    Returns (ktau_out, half_amplitude) computed via continuation in kτ.
    """
    N    = 2000
    phi  = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # precompute sin/cos for each harmonic order
    orders    = np.arange(1, n_harm + 1)           # 1, 2, ..., n_harm
    freq      = 2 * orders - 1                      # 1, 3, 5, ...
    signs     = (-1.0) ** (orders + 1)              # +1, -1, +1, ...
    sin_basis = np.sin(np.outer(freq, phi))         # (n_harm, N)
    cos_basis = np.cos(np.outer(freq, phi))         # (n_harm, N)

    def residual(R, ktau):
        delayed_arg = signs @ (R[:, None] * sin_basis)   # (N,)
        s = np.sin(delayed_arg)                           # (N,)
        # Vectorized computation instead of loop
        I_m = 2 * (s @ sin_basis.T) / N
        return R - (2 * ktau / (freq * np.pi)) * I_m

    HOPF = np.pi / 2
    ktau_out, amp_out = [], []
    R_prev = np.zeros(n_harm)

    for ktau in ktau_grid:
        if ktau <= HOPF:
            continue
        mu   = ktau - HOPF
        R0   = R_prev.copy()
        if R0[0] < 1e-3:
            R0[0] = 4 * np.sqrt(mu / np.pi)
        sol, _, ier, _ = fsolve(residual, R0, args=(ktau,), full_output=True)
        if ier == 1 and sol[0] > 1e-10:
            theta = cos_basis.T @ sol               # (N,)
            ktau_out.append(ktau)
            amp_out.append((theta.max() - theta.min()) / 2)
            R_prev = sol

    return np.array(ktau_out), np.array(amp_out)


def _decode_ic(s: str) -> float:
    return -float(s[1:]) if s.startswith("n") else float(s)


def parse_filename(path: str) -> dict | None:
    name = os.path.basename(path).removesuffix(".tsv")
    try:
        tau_str = name.split("tau_")[1].split("_k_")[0]
        k_str   = name.split("_k_")[1].split("_ic_")[0]
        ic_str  = name.split("_ic_")[1].split("_twarmup_")[0]
        return dict(tau=float(tau_str), k=float(k_str), ic=_decode_ic(ic_str))
    except (IndexError, ValueError):
        return None


def mean_amplitude_4tau(t: np.ndarray, theta: np.ndarray, tau: float) -> float:
    window = 4.0 * tau
    t0 = t[0]
    amplitudes = []
    idx = 0
    while True:
        t_start = t0 + idx * window
        t_end   = t_start + window
        if t_end > t[-1]:
            break
        seg = theta[(t >= t_start) & (t < t_end)]
        if len(seg) > 1:
            amplitudes.append(float((seg.max() - seg.min()) / 2))
        idx += 1
    return float(np.mean(amplitudes)) if amplitudes else float("nan")


def gather_timeseries_for_tauk(ts_cache: dict, target_tauk: float, tolerance: float = 0.01):
    """Gather all timeseries for a specific tau*k value from cache (within tolerance)."""
    all_t = []
    all_theta = []
    meta = None
    
    for tauk, entries in ts_cache.items():
        if abs(tauk - target_tauk) > tolerance:
            continue
        for t, theta, m in entries:
            all_t.append(t)
            all_theta.append(theta)
            meta = m
    
    return all_t, all_theta, meta["tau"] if meta else 1.0


def first_zero_crossing(t: np.ndarray, y: np.ndarray):
    """Return the first interpolated zero crossing time, or None if absent."""
    if t is None or y is None or len(t) < 2 or len(y) < 2:
        return None
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y0 == 0.0:
            return float(t0)
        if y1 == 0.0:
            return float(t1)
        if (y0 < 0.0 and y1 > 0.0) or (y0 > 0.0 and y1 < 0.0):
            dy = y1 - y0
            if dy == 0.0:
                continue
            frac = -y0 / dy
            return float(t0 + frac * (t1 - t0))
    return None


def plot_timeseries_inset(ax_main, ax_inset, all_t, all_theta, tau, tauk_label: str):
    """Plot a 4-tau window timeseries on an inset axes and return y-limits."""
    if not all_t:
        ax_inset.text(0.5, 0.5, "no data", transform=ax_inset.transAxes,
                      ha="center", va="center", fontsize=9)
        ax_inset.set_visible(True)
        return None
    
    # Use first timeseries, or average them
    t = all_t[0]
    theta = all_theta[0]
    
    # Show a 4-tau window starting at a zero crossing when available.
    window_size = 4.0 * tau
    t_zero = first_zero_crossing(t, theta)

    if t_zero is None:
        t_start = t[0] if len(t) > 0 else 0.0
        t_end = t_start + window_size
        mask = (t >= t_start) & (t <= t_end)
        t_seg = t[mask]
        y_seg = theta[mask]
        if len(t_seg) == 0:
            return None
        t_window = t_seg - t_seg[0]
        theta_window = y_seg
    else:
        mask = (t >= t_zero) & (t <= t_zero + window_size)
        t_seg = t[mask]
        y_seg = theta[mask]
        if len(t_seg) < 2:
            return None
        # Anchor the plotted segment exactly at (t=0, theta=0).
        t_seg = np.insert(t_seg, 0, t_zero)
        y_seg = np.insert(y_seg, 0, 0.0)
        t_window = t_seg - t_zero
        theta_window = y_seg
    
    if len(t_window) > 0:
        ax_inset.plot(t_window, theta_window, lw=1.0, color="#901A1E")
        ax_inset.set_xlim(0.0, window_size)
        y_pad = 0.08 * (np.max(theta_window) - np.min(theta_window) + 1e-12)
        ax_inset.set_ylim(np.min(theta_window) - y_pad, np.max(theta_window) + y_pad)
        # Keep tick positions (for grid lines) but hide marks/labels.
        ax_inset.set_xticks(np.linspace(0.0, window_size, 5))
        ax_inset.set_yticks(np.linspace(np.min(theta_window) - y_pad, np.max(theta_window) + y_pad, 5))
        ax_inset.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        ax_inset.set_xlabel(r"$t$", fontsize=16, labelpad=1)
        ax_inset.set_ylabel(r"$\theta$", fontsize=16, labelpad=2)
        ax_inset.grid(True, which="major", linewidth=0.45, alpha=0.5)
        return float(np.min(theta_window) - y_pad), float(np.max(theta_window) + y_pad)

    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts_dir     = os.path.join(script_dir, "outputs", "timeseries")
    plot_dir   = os.path.join(script_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Configure matplotlib to match paper_draft style
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 22,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    points: dict[float, list[float]] = {}
    ts_cache: dict[float, list] = {}  # Cache: tauk -> [(t, theta, meta), ...]

    for path in sorted(glob.glob(os.path.join(ts_dir, "*.tsv"))):
        meta = parse_filename(path)
        if meta is None or round(meta["ic"], 4) not in TARGET_ICS:
            continue

        try:
            data = np.loadtxt(path, skiprows=1)
        except Exception:
            continue
        if data.ndim != 2 or data.shape[0] < 2:
            continue

        t     = data[:, 0]
        theta = data[:, 1]

        n_drop = max(1, len(t) // 5)
        t     = t[n_drop:]
        theta = theta[n_drop:]

        tauk = round(meta["tau"] * meta["k"], 8)
        amp  = mean_amplitude_4tau(t, theta, meta["tau"])
        if not np.isnan(amp):
            points.setdefault(tauk, []).append(amp)
        
        # Cache for later use in insets
        ts_cache.setdefault(tauk, []).append((t, theta, meta))

    if not points:
        raise RuntimeError(
            f"No matching timeseries files found in {ts_dir!r}.\n"
            "Run the timeseries generator with IC ±1 first."
        )

    tauk_arr = np.array(sorted(points.keys()))
    amp_arr  = np.array([np.mean(points[tk]) for tk in tauk_arr])

    # ── Harmonic-balance (Bessel) theoretical curve with insets ───────────────
    # Self-consistency: 2 J1(R) / R = π / (2 τk)
    # Rearranged parametrically: τk(R) = π R / (4 J1(R))
    # amplitude (peak-to-peak) = 2R
    # Valid for R ∈ (0, j_{1,1}) where j_{1,1} ≈ 3.8317 is the first zero of J1.
    J1_ZERO = 3.8317  # first zero of J1
    R_vals  = np.linspace(1e-6, J1_ZERO - 1e-4, 2000)
    tauk_theory = np.pi * R_vals / (4.0 * j1(R_vals))
    amp_theory  = R_vals   # half-amplitude = R directly

    ktau_hb_grid       = np.linspace(np.pi / 2 + 0.02, tauk_arr.max(), 300)
    ktau_hb2, amp_hb2  = hb_nharmonic(ktau_hb_grid, n_harm=2)
    ktau_hb3, amp_hb3  = hb_nharmonic(ktau_hb_grid, n_harm=3)

    max_common = float(np.max(tauk_arr))
    if len(ktau_hb2) > 0:
        max_common = min(max_common, float(np.max(ktau_hb2)))
    if len(ktau_hb3) > 0:
        max_common = min(max_common, float(np.max(ktau_hb3)))
    theory_mask = (tauk_theory >= float(np.min(tauk_arr))) & (tauk_theory <= max_common)

    fig3, ax3 = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    fig3.patch.set_alpha(0.0)
    
    ax3.plot(tauk_arr, amp_arr, "o", ms=4, color="#901A1E", alpha=1, zorder=3,
             label=r"Simulation  (IC = $\pm 1$)")
    ax3.plot(tauk_theory[theory_mask], amp_theory[theory_mask], color="#547AA5", lw=2.0, ls="--", alpha=1, zorder=4,
             label=r"1-harmonic: $2J_1(R)/R = \pi/(2\tau k)$")
    ax3.plot(ktau_hb2, amp_hb2, color="#CBA810", lw=2.0, ls="--", alpha=1, zorder=5,
             label=r"2-harmonic HB: $+\, R_3\cos 3\omega t$")
    ax3.plot(ktau_hb3, amp_hb3, color="#A54E8E", lw=2.0, ls="--", alpha=1, zorder=6,
             label=r"3-harmonic HB: $+\, R_5\cos 5\omega t$")
    hopf_x = np.pi / 2
    ax3.plot([hopf_x, hopf_x], [0.0, 2.0], color="#666666", lw=1, ls="--", alpha=0.6,
             label=r"Hopf: $\tau k = \pi/2$")
    ax3.set_xlabel(r"$\tau k$", fontsize=32, labelpad=-4)
    ax3.set_ylabel(r"Average amplitude", fontsize=32, labelpad=-16)
    ax3.legend(fontsize=16, loc="upper left")
    ax3.set_xlim(float(np.min(tauk_arr)), tauk_arr.max() * 1.05)
    ax3.set_facecolor("none")
    ax3.grid(True, which="major", alpha=0.3)

    def y_label_formatter(val, _):
        if np.isclose(val, 0.0, rtol=0.0, atol=1e-10):
            return "0"
        if np.isclose(val, 3.0, rtol=0.0, atol=1e-10):
            return "3"
        return ""

    ax3.yaxis.set_major_formatter(FuncFormatter(y_label_formatter))
    
    # Style the main axes
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    
    ax3.minorticks_on()
    ax3.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=1)
    
    # Add insets for tau*k = 2 and tau*k = 4
    inset_tauk_values = [2.0, 4.0]
    xmin = float(np.min(tauk_arr))
    xmax = float(np.max(tauk_arr) * 1.05)
    inset_scale = 1.5
    inset_w, inset_h = 0.18 * inset_scale, 0.23 * inset_scale
    bottom = 0.17

    def x_center_to_left(x_center):
        frac = (x_center - xmin) / (xmax - xmin)
        return np.clip(frac - inset_w / 2.0, 0.02, 0.98 - inset_w)

    inset_positions = [
        [float(x_center_to_left(3.0)), bottom, inset_w, inset_h],
        [float(x_center_to_left(4.5)), bottom, inset_w, inset_h],
    ]
    
    inset_axes = []
    inset_ylims = []

    for target_tauk, inset_pos in zip(inset_tauk_values, inset_positions):
        all_t, all_theta, tau = gather_timeseries_for_tauk(ts_cache, target_tauk, tolerance=0.15)
        
        # Create inset axis
        ax_inset = fig3.add_axes(inset_pos)
        # Opaque inset background hides the main grid underneath.
        ax_inset.set_facecolor("white")
        
        y_limits = plot_timeseries_inset(ax3, ax_inset, all_t, all_theta, tau,
                                         rf"$\tau k = {target_tauk:.1f}$")
        if y_limits is not None:
            inset_ylims.append(y_limits)
        
        # Style inset
        for spine in ax_inset.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.7)
        ax_inset.minorticks_on()
        inset_axes.append(ax_inset)

    if inset_ylims:
        y_min_shared = min(v[0] for v in inset_ylims)
        y_max_shared = max(v[1] for v in inset_ylims)
        for ax_inset in inset_axes:
            ax_inset.set_ylim(y_min_shared, y_max_shared)
            ax_inset.set_yticks(np.linspace(y_min_shared, y_max_shared, 5))
    
    out_path3 = os.path.join(plot_dir, "amplitude_vs_tauK.svg")
    fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
    # fig3.savefig(out_path3.replace(".svg", ".pdf"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path3}")


if __name__ == "__main__":
    main()
