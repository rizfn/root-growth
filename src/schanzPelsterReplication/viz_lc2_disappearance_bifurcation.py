import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU   = 1.0
DT    = 0.01
N_BUF = round(TAU / DT) + 1   # 101

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _f(td, k):
    return -k * np.sin(td)


def spinup_dde(k, ic, t_warmup):
    buf   = np.full(N_BUF, float(ic)) if np.isscalar(ic) else np.array(ic, dtype=float)
    theta = buf[-1]
    for _ in range(int(t_warmup / DT)):
        k1     = _f(buf[0], k)
        k2     = _f(buf[1], k)
        theta  = theta + 0.5 * (k1 + k2) * DT
        buf[:-1] = buf[1:]
        buf[-1]  = theta
    return buf


def collect_chaotic_ics(k, t_warmup, n_samples, spacing_tau=1.0):
    """Spin up to the chaotic attractor at k, then snapshot n_samples
    buffer states spaced spacing_tau apart."""
    spacing_steps = max(1, round(spacing_tau * TAU / DT))
    buf   = spinup_dde(k, 2.0, t_warmup)
    theta = buf[-1]
    samples = []
    for _ in range(n_samples):
        samples.append(buf.copy())
        for _ in range(spacing_steps):
            k1     = _f(buf[0], k)
            k2     = _f(buf[1], k)
            theta  = theta + 0.5 * (k1 + k2) * DT
            buf[:-1] = buf[1:]
            buf[-1]  = theta
    return samples


def escape_time_tau(k, buf_init, win_tau=6.0, t_max_tau=5000.0):
    """Time (in τ) until max(θ) over a win_tau-wide window first drops below π.
    Returns t_max_tau if not escaped within t_max_tau.

    Same coarse-then-refine logic as viz_lc2_bifurcation_type.py:
    coarse pass checks every win_tau window; on trigger, suffix/prefix maxima
    give single-step resolution.
    """
    buf   = buf_init.copy()
    theta = buf[-1]
    win   = round(win_tau * TAU / DT)
    t_max = int(t_max_tau * TAU / DT)
    prev_win = np.full(win, theta)
    curr_win = np.empty(win)

    for step in range(t_max):
        k1     = _f(buf[0], k)
        k2     = _f(buf[1], k)
        theta  = theta + 0.5 * (k1 + k2) * DT
        buf[:-1] = buf[1:]
        buf[-1]  = theta
        curr_win[step % win] = theta

        if (step + 1) % win == 0:
            if curr_win.max() < np.pi:
                t_win_start = (step + 1 - win) * DT
                suffix_max  = np.maximum.accumulate(prev_win[::-1])[::-1]
                prefix_max  = np.maximum.accumulate(curr_win)
                combined    = np.empty(win)
                combined[:-1] = np.maximum(suffix_max[1:], prefix_max[:-1])
                combined[-1]  = prefix_max[-1]
                first_j = int(np.argmax(combined < np.pi))
                return (t_win_start + (first_j + 1) * DT) / TAU
            prev_win = curr_win.copy()
            curr_win = np.empty(win)
    return t_max_tau


def main():
    # ── Parameters ────────────────────────────────────────────────────────────
    k_ic        = 4.24    # IC source: stable chaotic LC2 attractor (below k_c)
    k_c         = 4.2404  # estimated critical point (between 4.2400 and 4.2405)
    t_warmup    = 50_000.0
    t_max       = 10_000.0
    n_samples   = 80      # independent ICs per k value
    spacing_tau = 1.0     # gap between collected ICs (in τ)

    # Scan k values above k_c: logarithmically spaced in δ = k - k_c
    esc_scan = k_c + np.geomspace(1e-4, 0.11, 40)

    # ── Collect ICs from chaotic attractor ───────────────────────────────────
    print(f"Collecting {n_samples} chaotic ICs at k = {k_ic}  "
          f"(spacing = {spacing_tau} τ,  warmup = {t_warmup:.0f})")
    ic_samples = collect_chaotic_ics(k_ic, t_warmup, n_samples, spacing_tau)
    print(f"  θ(samples 0–4) = {[f'{b[-1]:.3f}' for b in ic_samples[:5]]}")

    # ── Escape-time scan ──────────────────────────────────────────────────────
    print(f"\nEscape scan: {len(esc_scan)} k values × {n_samples} samples, "
          f"t_max = {t_max:.0f} τ")
    all_t_esc = []
    for k in esc_scan:
        times = np.array([escape_time_tau(k, buf, t_max_tau=t_max) for buf in ic_samples])
        all_t_esc.append(times)
        n_esc = (times < t_max).sum()
        med   = np.mean(times[times < t_max]) if n_esc else float("nan")
        cen   = n_samples - n_esc
        print(f"  k = {k:.4f}  mean T_esc = {med:8.1f} τ  "
              f"({n_esc}/{n_samples} escaped"
              + (f",  {cen} censored)" if cen else ")"))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    delta_arr = esc_scan - k_c    # positive = above k_c (ghost region)

    # Individual sample points
    first_dot, first_cen = True, True
    for delta, times in zip(delta_arr, all_t_esc):
        if delta <= 0:
            continue
        escaped = times < t_max
        if escaped.any():
            ax.scatter(np.full(escaped.sum(), delta), times[escaped],
                       color="#2166AC", s=15, alpha=0.15, zorder=3,
                       label="individual $T_{esc}$" if first_dot else "_")
            first_dot = False
        if (~escaped).any():
            ax.scatter(np.full((~escaped).sum(), delta),
                       np.full((~escaped).sum(), t_max * 0.85),
                       color="#2166AC", s=15, marker="v", alpha=0.15, zorder=3,
                       label=rf"censored ($T_{{esc}} > {t_max:.0f}\,\tau$)" if first_cen else "_")
            first_cen = False

    # mean per k (require ≥ 2 escaped samples)
    d_med, t_med = [], []
    for delta, times in zip(delta_arr, all_t_esc):
        if delta <= 0:
            continue
        esc = times[times < t_max]
        if len(esc) >= 2:
            d_med.append(delta)
            t_med.append(np.mean(esc))
    d_med = np.array(d_med)
    t_med = np.array(t_med)

    if len(d_med):
        ax.scatter(d_med, t_med, color="orange", s=90, zorder=5,
                   edgecolors="darkorange", linewidths=0.8,
                   label="mean $T_{esc}$ (escaped samples)")

    # Saddle-node reference power law + measured slope
    if len(d_med) >= 3:
        log_C = float(np.mean(np.log(t_med) + 0.5 * np.log(d_med)))
        C_fit = np.exp(log_C)
        d_fit = np.geomspace(d_med.min() * 0.4, d_med.max() * 2.0, 300)
        ax.plot(d_fit, C_fit * d_fit**(-0.5), color="#D6604D", lw=2.0, ls="--", zorder=4,
                label=rf"saddle-node ref: $C/\sqrt{{|\Delta k|}}$,  $C = {C_fit:.1f}\,\tau$")
        slope, log_C_fit = np.polyfit(np.log(d_med), np.log(t_med), 1)
        C_meas = np.exp(log_C_fit)
        ax.plot(d_fit, C_meas * d_fit**slope, color="#4DAC26", lw=2.0, ls="-", zorder=4,
                label=rf"fitted: $C\,|\Delta k|^{{{slope:.2f}}}$,  $C = {C_meas:.1f}\,\tau$")
        ax.annotate(
            rf"measured slope (mean): ${slope:.2f}$   (saddle-node: $-0.50$)",
            xy=(0.04, 0.08), xycoords="axes fraction", fontsize=10, color="#2166AC",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2166AC", alpha=0.8),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"$k - k_c$,   $k_c \approx {k_c}$", fontsize=12)
    ax.set_ylabel(r"$T_{esc}\,/\,\tau$", fontsize=12)
    ax.set_title(
        rf"Ghost escape time near disappearance of chaotic LC2  "
        rf"($\tau = 1$,  {n_samples} ICs from chaotic attractor at $k = {k_ic}$)"
        "\n"
        rf"$\mathrm{{d}}\theta/\mathrm{{d}}t = -k\sin(\theta(t-\tau))$,   "
        rf"$k_c \approx {k_c}$ (between 4.2400 and 4.2405)"
        "\n"
        r"Saddle-node bifurcation: slope $= -1/2$.   "
        r"Crisis / attractor collision: different power-law exponent.",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    out = os.path.join(SCRIPT_DIR, "plots", "transients", "lc2_disappearance_bifurcation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
