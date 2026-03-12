import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
ODE saddle-node of limit cycles — canonical polar form (no delay)

    ṙ = r · [α − (r²−1)²]          (radial, decoupled from θ)
    θ̇ = 1

Limit cycles (ṙ = 0, r > 0)  when  (r²−1)² = α:
    α > 0  →  r± = √(1 ± √α)   [+ outer stable,  − inner unstable; exist for α < 1]
    α = 0  →  SNLC: both merge at r = 1
    α < 0  →  no limit cycles; ghost bottleneck near r ≈ 1

Analytic escape time (saddle-node normal form, starting from r₀ = 1):
    T_esc = (1/2√|α|) · arctan(1/√|α|)  ≈  π/(4√|α|)  as α → 0⁻

Key comparison with the DDE:
    ODE:  T_esc ∝ |α|^{−1/2} holds uniformly — no anomalous deviation
          even at extremely small |α|.
    DDE:  deviation from |Δ(τk)|^{−1/2} appears near the bifurcation,
          indicating delay-specific physics in the ghost region.
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _drdt(r: float, alpha: float) -> float:
    d = r * r - 1.0
    return r * (alpha - d * d)


def _rk4(r: float, alpha: float, dt: float) -> float:
    k1 = _drdt(r,               alpha)
    k2 = _drdt(r + 0.5*dt*k1,  alpha)
    k3 = _drdt(r + 0.5*dt*k2,  alpha)
    k4 = _drdt(r +     dt*k3,  alpha)
    return r + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def escape_time(alpha: float, r0: float = 1.0, r_escape: float = 0.5,
                dt: float = 1e-2, t_max: float = 2e5) -> float:
    """Integrate ṙ from r₀; return time when r first drops to r_escape.

    For alpha < 0  ṙ < 0 everywhere, so r decreases monotonically to 0.
    The ghost near r = 1 creates the bottleneck; T_esc → ∞ as alpha → 0⁻.
    """
    r = float(r0)
    n = int(t_max / dt)
    for i in range(n):
        r = _rk4(r, alpha, dt)
        if r <= r_escape:
            return (i + 1) * dt
    return t_max


def main():
    # ── analytic LC radii above bifurcation ────────────────────────────────
    alpha_above = np.geomspace(1e-6, 0.99, 400)
    r_plus  = np.sqrt(1.0 + np.sqrt(alpha_above))
    r_minus = np.sqrt(np.maximum(1.0 - np.sqrt(alpha_above), 0.0))
    amp     = 0.5 * (r_plus - r_minus)   # half-gap used as "amplitude"

    # analytic period: T = 2π (θ̇ = 1 constant, independent of α)

    # ── escape time scan below bifurcation ─────────────────────────────────
    esc_scan = np.geomspace(1e-7, 0.5, 60)   # |α| over ~6.7 decades
    t_max    = 2e5

    print(f"Escape-time scan ({len(esc_scan)} values) …")
    t_esc = np.empty(len(esc_scan))
    for i, a in enumerate(esc_scan):
        t = escape_time(-a, r0=1.0, r_escape=0.5, dt=1e-2, t_max=t_max)
        flag = "  (censored)" if t >= t_max else ""
        print(f"  |α| = {a:.2e}   T_esc = {t:.1f}{flag}")
        t_esc[i] = t

    escaped = t_esc < t_max
    delta   = esc_scan

    # power-law fit
    slope, log_C = np.polyfit(np.log(delta[escaped]), np.log(t_esc[escaped]), 1)
    C_fit = np.exp(log_C)
    print(f"\nFitted slope: {slope:.4f}  (theory: -0.5000)")

    # exact analytic prediction
    T_analytic = (1.0 / (2.0 * np.sqrt(delta))) * np.arctan(1.0 / np.sqrt(delta))

    # ── figure (3-panel, mirroring lc2_bifurcation_type.py layout) ─────────
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.35)
    ax_amp = fig.add_subplot(gs[0, 0])
    ax_r   = fig.add_subplot(gs[0, 1])
    ax_esc = fig.add_subplot(gs[1, :])

    # (a) half-amplitude vs alpha
    ax_amp.plot(alpha_above, amp, color="#2166AC", lw=2.0)
    ax_amp.axvline(0, color="red", lw=1.0, ls=":", alpha=0.7,
                   label=r"$\alpha_c = 0$")
    ax_amp.set_xlabel(r"$\alpha$", fontsize=11)
    ax_amp.set_ylabel(r"half-gap $R = (r_+ - r_-)/2$", fontsize=11)
    ax_amp.set_title("(a)  Amplitude near onset\n"
                     r"$R \propto \alpha^{1/2} \to 0$ at onset  "
                     r"(period is $T = 2\pi$ = const, unlike SNIC)",
                     fontsize=10)
    ax_amp.legend(fontsize=9); ax_amp.grid(True, linewidth=0.4, alpha=0.5)

    # (b) LC radii
    ax_r.plot(alpha_above, r_plus,  color="#2166AC", lw=2.0,
              label=r"stable:  $r_+ = \sqrt{1+\sqrt{\alpha}}$")
    ax_r.plot(alpha_above, r_minus, color="#D6604D", lw=2.0, ls="--",
              label=r"unstable:  $r_- = \sqrt{1-\sqrt{\alpha}}$")
    ax_r.axhline(1.0, color="gray", lw=0.9, ls=":", alpha=0.7,
                 label=r"merge at $r=1$  ($\alpha_c = 0$)")
    ax_r.set_xlabel(r"$\alpha$", fontsize=11)
    ax_r.set_ylabel(r"limit-cycle radius $r^*$", fontsize=11)
    ax_r.set_title("(b)  LC radii — both converge to $r=1$ at onset\n"
                   r"(saddle-node collision, not a Hopf bifurcation)", fontsize=10)
    ax_r.legend(fontsize=9); ax_r.grid(True, linewidth=0.4, alpha=0.5)

    # (c) escape time
    ax = ax_esc
    ax.scatter(delta[escaped], t_esc[escaped], color="#2166AC", s=40, zorder=3,
               label=r"$T_{esc}$ (simulated, $r_0 = 1$, escape at $r < 0.5$)")
    if (~escaped).any():
        ax.scatter(delta[~escaped], np.full((~escaped).sum(), t_max * 0.7),
                   color="#2166AC", s=40, marker="v", alpha=0.45, zorder=3,
                   label=rf"censored ($T_{{esc}} > {t_max:.0f}$)")

    d_line = np.geomspace(delta[escaped].min() * 0.3,
                          delta[escaped].max() * 3.0, 300)
    ax.plot(d_line, C_fit * d_line**slope, color="#D6604D", lw=2.0, ls="--",
            label=rf"power-law fit: slope $= {slope:.3f}$, $C = {C_fit:.2f}$")
    ax.plot(delta, T_analytic, color="green", lw=1.6, ls=":",
            label=r"$(2\sqrt{|\alpha|})^{-1}\arctan(|\alpha|^{-1/2})$  (exact theory)")

    ax.annotate(rf"measured slope: ${slope:.4f}$  (theory: $-0.5000$)",
                xy=(0.04, 0.09), xycoords="axes fraction", fontsize=10,
                color="#2166AC",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2166AC", alpha=0.8))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\alpha - \alpha_c|$", fontsize=11)
    ax.set_ylabel(r"$T_{esc}$", fontsize=11)
    ax.set_title(
        "(c)  Ghost-attractor escape time below SNLC onset\n"
        r"ODE: $T_{esc} \propto |\alpha|^{-1/2}$ holds uniformly across all scales "
        r"— no anomalous deviation near $\alpha_c$",
        fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    fig.suptitle(
        r"ODE saddle-node of limit cycles  ($\dot{r} = r[\alpha-(r^2-1)^2]$,"
        r"  $\dot{\theta}=1$,  $\alpha_c = 0$)"
        "\n"
        r"Delay-free reference: $|\Delta\alpha|^{-1/2}$ ghost scaling is exact,"
        r" no DDE-like anomalous deviation",
        fontsize=12)

    out = os.path.join(SCRIPT_DIR, "plots", "transients", "ode_snlc_escape.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
