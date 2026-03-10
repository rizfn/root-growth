"""
viz_transient_second_lc.py
==========================
Transient analysis at the two boundaries of the second limit cycle in
    dθ/dt = -k·sin(θ(t-τ))   (τ = 25)

Workflow
--------
1. Spin up at τ·k = TAUK_REF  (IC = IC_REF) until the 2nd LC is established.
   This gives the reference LC2 steady-state history (delay-window buffer).

2. Use that buffer as the IC for two transient runs, each T_TRANS long:
   Case A  τ·k = TAUK_A  — just *before* LC2 comes into existence.
   Case B  τ·k = TAUK_B  — just *after* LC2 ceases to exist.

Output: plots/transients/transient_ref{TAUK_REF}_A{TAUK_A}_B{TAUK_B}.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
TAU         = 25.0
DT          = 0.01
DELAY_STEPS = round(TAU / DT)   # buffer length = τ / dt

# ── Spin-up parameters ────────────────────────────────────────────────────────
TAUK_REF = 4.105   # τ·k used to establish the LC2 reference state
IC_REF   = 2.0     # constant-history IC for the spin-up
T_WARMUP = 100_000.0
# Old values (previous version):
# TAUK_REF = 4.11 ; IC_REF = 2.0 ; T_WARMUP = 100_000.0

# ── Transient parameters ──────────────────────────────────────────────────────
TAUK_A  = 3.9849   # Case A: just before LC2 onset
TAUK_B  = 4.240   # Case B: just after LC2 end
T_TRANS = 1000.0 * TAU   # 1000τ recording window
RECORD_DT = 0.1
# Old values (previous version):
# TAUK_A = 4.104 ; TAUK_B = 4.241 ; T_TRANS = 30_000.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR   = os.path.join(SCRIPT_DIR, "plots", "transients")
os.makedirs(PLOT_DIR, exist_ok=True)


def _f(theta_delayed: float, k: float) -> float:
    return -k * np.sin(theta_delayed)


def spinup_dde(k: float, ic, t_warmup: float) -> np.ndarray:
    """
    Integrate the DDE for t_warmup, return the final delay buffer.

    ic  – scalar float (constant history) or ndarray of shape (DELAY_STEPS+1,).
    Returns buf of shape (DELAY_STEPS+1,), oldest … newest.
    """
    buf = np.full(DELAY_STEPS + 1, float(ic)) if np.isscalar(ic) \
          else np.array(ic, dtype=float)
    theta = buf[-1]

    for _ in range(int(t_warmup / DT)):
        k1    = _f(buf[0], k)
        k2    = _f(buf[1], k)
        theta = theta + 0.5 * (k1 + k2) * DT
        buf[:-1] = buf[1:]
        buf[-1]  = theta

    return buf


def integrate_dde(k: float, history: np.ndarray,
                  t_measure: float, record_dt: float = 0.1):
    """
    Integrate the DDE from the given history for t_measure time units.
    Returns (t_arr, theta_arr) sampled every record_dt.
    """
    buf   = history.copy().astype(float)
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
        k1    = _f(buf[0], k)
        k2    = _f(buf[1], k)
        theta = theta + 0.5 * (k1 + k2) * DT
        buf[:-1] = buf[1:]
        buf[-1]  = theta
        if (step + 1) % rec_stride == 0 and out_idx < n_out:
            t_arr[out_idx]     = (step + 1) * DT
            theta_arr[out_idx] = theta
            out_idx += 1

    return t_arr[:out_idx], theta_arr[:out_idx]


def run_all():
    # ── Reference: establish LC2 at τ·k = TAUK_REF ───────────────────────────
    k_ref = TAUK_REF / TAU
    print(f"Spinning up LC2 reference at τ·k = {TAUK_REF}  (k = {k_ref:.8f})")
    print(f"  IC = {IC_REF},  t_warmup = {T_WARMUP:.0f} …")
    buf_ref = spinup_dde(k_ref, ic=IC_REF, t_warmup=T_WARMUP)
    print(f"  Done.  θ(end) = {buf_ref[-1]:.4f}")

    # ── Case A: τ·k = 4.101 (before LC2 onset) ──────────────────────────────
    k_a = TAUK_A / TAU
    print(f"\nCase A: transient at τ·k = {TAUK_A}  for {T_TRANS / TAU:.0f}τ = {T_TRANS:.0f} …")
    t_a, th_a = integrate_dde(k_a, buf_ref, t_measure=T_TRANS, record_dt=RECORD_DT)
    print(f"  Done.  {len(t_a)} points recorded.")

    # ── Case B: τ·k = 4.242 (after LC2 end) ─────────────────────────────────
    k_b = TAUK_B / TAU
    print(f"\nCase B: transient at τ·k = {TAUK_B}  for {T_TRANS / TAU:.0f}τ = {T_TRANS:.0f} …")
    t_b, th_b = integrate_dde(k_b, buf_ref, t_measure=T_TRANS, record_dt=RECORD_DT)
    print(f"  Done.  {len(t_b)} points recorded.")

    return buf_ref, t_a, th_a, t_b, th_b


def plot_all(buf_ref, t_a, th_a, t_b, th_b):
    fig, axes = plt.subplots(3, 1, figsize=(13, 14),
                             gridspec_kw={"hspace": 0.45})

    # ── Panel 1: LC2 steady-state IC (the delay-window history) ─────────────
    ax = axes[0]
    t_buf = np.linspace(-TAU, 0, DELAY_STEPS + 1)
    ax.plot(t_buf, buf_ref, color="#2166AC", lw=1.0)
    ax.axhline( np.pi, color="gray", lw=0.8, ls="--", alpha=0.6, label=r"$\pm\pi$")
    ax.axhline(-np.pi, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(
        rf"LC2 steady-state IC  ($\tau k = {TAUK_REF}$,  warmup = {T_WARMUP:.0f})"
        "\ndelay-window history  θ(t − τ … t)",
        fontsize=11,
    )
    ax.set_xlabel(r"relative time  (units of $\tau$)", fontsize=10)
    ax.set_ylabel(r"$\theta$", fontsize=11)
    ax.set_xticks(np.linspace(-TAU, 0, 6))
    ax.set_xticklabels([rf"${x/TAU:.1f}\tau$" for x in np.linspace(-TAU, 0, 6)])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── Panel 2: Case A  τ·k = 4.101 ────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_a / TAU, th_a, color="#2166AC", lw=0.7)
    ax.axhline( np.pi, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-np.pi, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(
        rf"Case A  ($\tau k = {TAUK_A}$, just *before* LC2 onset)"
        "\nIC = LC2 steady state from "
        rf"$\tau k = {TAUK_REF}$",
        fontsize=11,
    )
    ax.set_xlabel(r"$t\,/\,\tau$", fontsize=10)
    ax.set_ylabel(r"$\theta(t)$", fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── Panel 3: Case B  τ·k = 4.242 ────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_b / TAU, th_b, color="#D6604D", lw=0.7)
    ax.axhline( np.pi, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-np.pi, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(
        rf"Case B  ($\tau k = {TAUK_B}$, just *after* LC2 end)"
        "\nIC = LC2 steady state from "
        rf"$\tau k = {TAUK_REF}$",
        fontsize=11,
    )
    ax.set_xlabel(r"$t\,/\,\tau$", fontsize=10)
    ax.set_ylabel(r"$\theta(t)$", fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        r"Transient decay of the second limit cycle  ($\tau = 25$)"
        "\n"
        r"$\mathrm{d}\theta/\mathrm{d}t = -k\sin(\theta(t-\tau))$",
        fontsize=13,
    )

    fname = f"transient_ref{TAUK_REF}_A{TAUK_A}_B{TAUK_B}.png"
    out = os.path.join(PLOT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


def main():
    buf_ref, t_a, th_a, t_b, th_b = run_all()
    plot_all(buf_ref, t_a, th_a, t_b, th_b)
    print("All done.")


if __name__ == "__main__":
    main()
