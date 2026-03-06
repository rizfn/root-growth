"""
viz_transient_second_lc.py
==========================
Transient analysis for two bifurcation events in
    dθ/dt = -k·sin(θ(t-τ))   (τ = 25)

Case A – onset of the second limit cycle at τ·k ≈ 4.105
    1. Spin up at τ·k = 4.106  (IC = 2.0)  until the 2nd LC is established.
    2. Use the last τ seconds of that trajectory as the delay-history IC.
    3. Run at τ·k = 4.104  (just *below* the bifurcation) and record the
       transient: the new LC is unstable there, so θ(t) drifts back to
       the original attractor.

Case B – end of the chaotic window at τ·k ≈ 4.24
    1. Spin up at τ·k = 4.24  (IC = 2.0)  to get a chaotic history.
    2. Use that as the delay-history IC.
    3. Run at τ·k = 4.241  (just *after* chaos disappears) and record the
       transient: the chaotic state collapses onto the surviving LC.

Additionally:
    - Plot the limit-cycle IC used for Case A (the 4.106 attractor).
    - Plot the chaotic IC used for Case B (the 4.24 attractor).
    - Plot both transient time series.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
TAU   = 25.0
DT    = 0.01          # integration step
DELAY_STEPS = round(TAU / DT)   # integer index for exact delay lookup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR   = os.path.join(SCRIPT_DIR, "plots", "transients")
os.makedirs(PLOT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Integrator
# ═══════════════════════════════════════════════════════════════════════════════

def _dtheta(theta_delayed: float, k: float) -> float:
    return -k * np.sin(theta_delayed)


def integrate_dde(
    k: float,
    history: np.ndarray,       # shape (DELAY_STEPS+1,) – most-recent last
    t_measure: float,
    record_dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the DDE for t_measure time units using Heun's method.

    The delay buffer is kept as a circular array of length DELAY_STEPS+1.
    history[-1] is the current state; history[0] is θ(t - τ).

    Returns (t_arr, theta_arr) at record_dt spacing.
    """
    buf = history.copy().astype(float)   # circular window, oldest … newest
    theta = buf[-1]

    rec_stride = max(1, round(record_dt / DT))
    n_steps    = int(t_measure / DT)
    n_out      = n_steps // rec_stride + 1

    t_arr     = np.empty(n_out)
    theta_arr = np.empty(n_out)

    t_arr[0]     = 0.0
    theta_arr[0] = theta

    out_idx = 1

    for step in range(n_steps):
        theta_d  = buf[0]                   # θ(t − τ)
        k1       = _dtheta(theta_d, k)
        theta_p  = theta + k1 * DT

        # For Heun: need θ(t+dt − τ).  The buffer shifted by 1 step forward
        # means the delayed value is buf[1] (the next oldest in the queue).
        theta_d2 = buf[1]
        k2       = _dtheta(theta_d2, k)
        theta    = theta + 0.5 * (k1 + k2) * DT

        # Shift circular buffer: drop oldest, append new
        buf[:-1] = buf[1:]
        buf[-1]  = theta

        if (step + 1) % rec_stride == 0 and out_idx < n_out:
            t_arr[out_idx]     = (step + 1) * DT
            theta_arr[out_idx] = theta
            out_idx += 1

    return t_arr[:out_idx], theta_arr[:out_idx]


def spinup_dde(
    k: float,
    ic: "float | np.ndarray",
    t_warmup: float,
) -> np.ndarray:
    """
    Integrate the DDE for t_warmup.

    ic can be:
      - a scalar float  → constant history θ(t) = ic for t ∈ [-τ, 0]
      - an ndarray of shape (DELAY_STEPS+1,)  → use directly as initial history

    Returns the final delay-buffer (length DELAY_STEPS+1), newest last.
    """
    if isinstance(ic, np.ndarray):
        buf   = ic.copy().astype(float)
    else:
        buf   = np.full(DELAY_STEPS + 1, ic, dtype=float)
    theta = buf[-1]

    n_steps = int(t_warmup / DT)
    for step in range(n_steps):
        theta_d  = buf[0]
        k1       = _dtheta(theta_d, k)
        theta_p  = theta + k1 * DT

        theta_d2 = buf[1]
        k2       = _dtheta(theta_d2, k)
        theta    = theta + 0.5 * (k1 + k2) * DT

        buf[:-1] = buf[1:]
        buf[-1]  = theta

    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# Case A – second LC onset
# ═══════════════════════════════════════════════════════════════════════════════

def case_a():
    """
    Step 1: spin up at τ·k = 4.11, IC = 2.0
    Step 2: run at τ·k = 4.104 using that history; record transient
    """
    print("\n── Case A: second LC onset (τ·k near 4.105) ──")

    # ── Step 1 : establish the second limit cycle at 4.11 ───────────────────
    k_lc    = 4.11 / TAU
    T_WARMUP_LC = 100_000.0      # long warmup to settle fully onto the 2nd LC
    print(f"  Spinning up at τ·k = 4.11  (k = {k_lc:.6f})  for t = {T_WARMUP_LC:.0f} …")
    buf_lc = spinup_dde(k_lc, ic=2.0, t_warmup=T_WARMUP_LC)
    print(f"  Done. θ(end) = {buf_lc[-1]:.4f}")

    # ── Step 2 : run transient at 4.104 ─────────────────────────────────────
    k_below = 4.104 / TAU
    T_TRANS  = 30_000.0
    print(f"  Running transient at τ·k = 4.104  (k = {k_below:.6f})  for t = {T_TRANS:.0f} …")
    t_a, th_a = integrate_dde(k_below, buf_lc, t_measure=T_TRANS, record_dt=0.1)
    print(f"  Done. {len(t_a)} time points recorded.")

    return buf_lc, t_a, th_a



# ═══════════════════════════════════════════════════════════════════════════════
# Case B – end of chaotic window
# ═══════════════════════════════════════════════════════════════════════════════

def case_b(buf_lc: np.ndarray):
    """
    Step 1: spin up at τ·k = 4.24, starting from the Case-A LC buffer,
            to get a chaotic history.
    Step 2: run at τ·k = 4.241 ; chaos cannot persist → transient to LC.
    Also return the full IC trajectory for an extra plot.
    """
    print("\n── Case B: end of chaos (τ·k near 4.24) ──")

    # ── Step 1 : chaotic state at 4.24, seeded with the Case-A LC ───────────
    k_chaos   = 4.24 / TAU
    T_WARMUP_CHAOS = 50_000.0
    print(f"  Spinning up at τ·k = 4.24  (k = {k_chaos:.6f})  for t = {T_WARMUP_CHAOS:.0f}")
    print(f"  (starting from Case-A LC history at τ·k = 4.11) …")
    buf_chaos = spinup_dde(k_chaos, ic=buf_lc, t_warmup=T_WARMUP_CHAOS)
    print(f"  Done. θ(end) = {buf_chaos[-1]:.4f}")

    # Record a few cycles of the chaotic IC for plotting
    print("  Recording chaotic IC trajectory for plot …")
    t_ic, th_ic = integrate_dde(k_chaos, buf_chaos, t_measure=2000.0, record_dt=0.1)

    # ── Step 2 : run transient at 4.241 ─────────────────────────────────────
    k_above  = 4.241 / TAU
    T_TRANS  = 30_000.0
    print(f"  Running transient at τ·k = 4.241  (k = {k_above:.6f})  for t = {T_TRANS:.0f} …")
    t_b, th_b = integrate_dde(k_above, buf_chaos, t_measure=T_TRANS, record_dt=0.1)
    print(f"  Done. {len(t_b)} time points recorded.")

    return buf_chaos, t_ic, th_ic, t_b, th_b


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(buf_lc, t_a, th_a,
             buf_chaos, t_ic, th_ic,
             t_b, th_b):

    fig, axes = plt.subplots(4, 1, figsize=(14, 18),
                             gridspec_kw={"hspace": 0.42})

    # ── Panel 1: IC used for Case A (the 4.106 LC) ──────────────────────────
    ax = axes[0]
    # The buf_lc is the delay window; plot it as a time trace
    t_buf  = np.linspace(-TAU, 0, DELAY_STEPS + 1)
    ax.plot(t_buf, buf_lc, color="#2166AC", lw=1.0)
    ax.set_title(r"Initial condition for Case A: limit cycle at $\tau k = 4.11$"
                 "\n(delay-window history = θ(t−τ … t)  after 100 000 warmup steps)",
                 fontsize=10)
    ax.set_xlabel("relative time (units of τ)", fontsize=9)
    ax.set_ylabel("θ", fontsize=10)
    ax.set_xticks(np.arange(-TAU, 1, 5))
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Transient at 4.104 ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_a, th_a, color="#2166AC", lw=0.7)
    ax.set_title(r"Case A transient: $\tau k = 4.104$ (IC = 4.11 LC)"
                 "\nSecond LC does not exist here → trajectory relaxes to "
                 "original attractor",
                 fontsize=10)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel("θ(t)", fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: IC used for Case B (chaotic orbit at 4.24) ─────────────────
    ax = axes[2]
    ax.plot(t_ic, th_ic, color="#D6604D", lw=0.8)
    ax.set_title(r"Initial condition for Case B: chaotic orbit at $\tau k = 4.24$"
                 "\n(seeded from Case-A LC at τ·k = 4.11, then run for 50 000 warmup)",
                 fontsize=10)
    ax.set_xlabel("t  (relative, after 50 000 warmup)", fontsize=9)
    ax.set_ylabel("θ(t)", fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Transient at 4.241 ──────────────────────────────────────────
    ax = axes[3]
    ax.plot(t_b, th_b, color="#D6604D", lw=0.7)
    ax.set_title(r"Case B transient: $\tau k = 4.241$ (IC = 4.24 chaos)"
                 "\nChaotic window has ended → trajectory settles onto surviving LC",
                 fontsize=10)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel("θ(t)", fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        r"DDE transients across two bifurcations:  d$\theta$/dt = $-k\sin(\theta(t-\tau))$,  $\tau = 25$",
        fontsize=12)

    out = os.path.join(PLOT_DIR, "transient_second_lc.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    buf_lc, t_a, th_a               = case_a()
    buf_chaos, t_ic, th_ic, t_b, th_b = case_b(buf_lc)
    plot_all(buf_lc, t_a, th_a, buf_chaos, t_ic, th_ic, t_b, th_b)
    print("\nAll done.")


if __name__ == "__main__":
    main()
