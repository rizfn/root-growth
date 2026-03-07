import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.special import j1

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
        out = []
        for m_idx, (f, R_m) in enumerate(zip(freq, R)):
            I_m = 2 * np.dot(s, sin_basis[m_idx]) / N
            out.append(R_m - (2 * ktau / (f * np.pi)) * I_m)
        return out

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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts_dir     = os.path.join(script_dir, "outputs", "timeseries")
    plot_dir   = os.path.join(script_dir, "plots", "amplitude")
    os.makedirs(plot_dir, exist_ok=True)

    points: dict[float, list[float]] = {}

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

    if not points:
        raise RuntimeError(
            f"No matching timeseries files found in {ts_dir!r}.\n"
            "Run './timeseries' with IC ±1 first."
        )

    tauk_arr = np.array(sorted(points.keys()))
    amp_arr  = np.array([np.mean(points[tk]) for tk in tauk_arr])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tauk_arr, amp_arr, s=8, color="#2166AC", alpha=0.85, zorder=3,
               label=r"IC = $\pm 1$")
    ax.set_xlabel(r"$\tau \cdot k$", fontsize=13)
    ax.set_ylabel(
        r"Average amplitude  $\langle\,(\max\theta - \min\theta)/2\,\rangle_{4\tau}$",
        fontsize=12,
    )
    ax.set_title(r"Oscillation amplitude vs $\tau k$", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, "amplitude_vs_tauK.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")

    # ── Fits ──────────────────────────────────────────────────────────────────
    HOPF = np.pi / 2
    mask = tauk_arr > HOPF
    x = tauk_arr[mask] - HOPF      # shifted so onset is at 0
    y = amp_arr[mask]

    fits = {
        r"$c\,({\tau k - \pi/2})^{\,\alpha}$": (
            lambda x, c, a: c * x**a,
            (1.0, 0.5),
            lambda p: rf"$c={p[0]:.2f},\ \alpha={p[1]:.2f}$",
        ),
        r"$A_\infty\tanh\!\left(b\,(\tau k-\pi/2)\right)$": (
            lambda x, A, b: A * np.tanh(b * x),
            (4.0, 0.5),
            lambda p: rf"$A_\infty={p[0]:.2f},\ b={p[1]:.2f}$",
        ),
        r"$A_\infty\tanh\!\left(b\sqrt{\tau k-\pi/2}\right)$": (
            lambda x, A, b: A * np.tanh(b * np.sqrt(x)),
            (4.0, 0.5),
            lambda p: rf"$A_\infty={p[0]:.2f},\ b={p[1]:.2f}$",
        ),
    }

    colors = ["#D6604D", "#1A9641", "#9970AB"]
    x_plot = np.linspace(0, tauk_arr.max() - HOPF, 500)

    fig2, axes = plt.subplots(1, len(fits), figsize=(5 * len(fits), 4.5), sharey=True)
    for ax2, (label, (fn, p0, fmt)), color in zip(axes, fits.items(), colors):
        ax2.scatter(tauk_arr, amp_arr, s=6, color="#2166AC", alpha=0.6,
                    zorder=3, label="data")
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=10000)
            ax2.plot(x_plot + HOPF, fn(x_plot, *popt), color=color,
                     lw=1.8, zorder=4, label=label + "\n" + fmt(popt))
        except RuntimeError:
            ax2.text(0.5, 0.5, "fit failed", transform=ax2.transAxes, ha="center")
        ax2.axvline(HOPF, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax2.set_xlabel(r"$\tau \cdot k$", fontsize=12)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel(r"Average amplitude", fontsize=12)
    fig2.suptitle(r"Amplitude fits vs $\tau k$", fontsize=13)
    plt.tight_layout()

    out_path2 = os.path.join(plot_dir, "amplitude_fits_tauK.png")
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved → {out_path2}")

    # ── Harmonic-balance (Bessel) theoretical curve ───────────────────────────
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

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(tauk_arr, amp_arr, s=8, color="#2166AC", alpha=0.85, zorder=3,
                label=r"simulation  (IC = $\pm 1$)")
    ax3.plot(tauk_theory, amp_theory, color="#D6604D", lw=2.0, zorder=4,
             label=r"1-harmonic HB: $2J_1(R)/R = \pi/(2\tau k)$")
    ax3.plot(ktau_hb2, amp_hb2, color="#1A9641", lw=2.0, ls="--", zorder=5,
             label=r"2-harmonic HB: $R_1\cos\omega t + R_3\cos 3\omega t$")
    ax3.plot(ktau_hb3, amp_hb3, color="#9970AB", lw=2.0, ls=":", zorder=6,
             label=r"3-harmonic HB: $+\, R_5\cos 5\omega t$")
    ax3.axvline(np.pi / 2, color="gray", lw=0.8, ls="--", alpha=0.6,
                label=r"Hopf: $\tau k = \pi/2$")
    ax3.set_xlabel(r"$\tau \cdot k$", fontsize=13)
    ax3.set_ylabel(r"Average amplitude", fontsize=12)
    ax3.set_title(r"Bessel self-consistency vs simulation", fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, linewidth=0.4, alpha=0.5)
    ax3.set_xlim(0, tauk_arr.max() * 1.05)
    plt.tight_layout()

    out_path3 = os.path.join(plot_dir, "amplitude_bessel_tauK.png")
    fig3.savefig(out_path3, dpi=150)
    print(f"Saved → {out_path3}")


if __name__ == "__main__":
    main()
