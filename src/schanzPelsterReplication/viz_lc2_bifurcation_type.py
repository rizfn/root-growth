import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

TAU = 25.0
DT  = 0.001
N_BUF = round(TAU / DT) + 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _f(td, k):
    return -k * np.sin(td)


def spinup_dde(k, ic, t_warmup):
    buf   = np.full(N_BUF, float(ic)) if np.isscalar(ic) else np.array(ic, dtype=float)
    theta = buf[-1]
    for _ in range(int(t_warmup / DT)):
        k1    = _f(buf[0], k)
        k2    = _f(buf[1], k)
        theta = theta + 0.5 * (k1 + k2) * DT
        buf[:-1] = buf[1:]
        buf[-1]  = theta
    return buf


def measure_amplitude_period(k, ic_buf, t_measure=5000.0, record_dt=0.1):
    buf, theta = ic_buf.copy(), ic_buf[-1]
    rec, pts = max(1, round(record_dt / DT)), []
    for step in range(int(t_measure / DT)):
        k1    = _f(buf[0], k)
        k2    = _f(buf[1], k)
        theta = theta + 0.5 * (k1 + k2) * DT
        buf[:-1] = buf[1:]
        buf[-1]  = theta
        if (step + 1) % rec == 0:
            pts.append(theta)
    arr = np.array(pts[len(pts) // 2:])
    amp = 0.5 * (arr.max() - arr.min())
    arr_c = arr - arr.mean()
    fft   = np.fft.rfft(arr_c, n=2 * len(arr_c))
    acf   = np.fft.irfft(fft * np.conj(fft))[:len(arr_c)].real
    if acf[0] > 0:
        acf /= acf[0]
        peaks, _ = find_peaks(acf, distance=max(1, int(1.5 * TAU / record_dt)), height=0.3)
        per = float(peaks[0]) * record_dt if len(peaks) else float("nan")
    else:
        per = float("nan")
    return amp, per


def escape_time_tau(k, buf_init, win_tau=6.0, t_max_tau=500.0):
    """Time (in τ) of first t s.t. max(θ[t-win_tau, t]) < π (step-resolution).

    Coarse pass: check every win_tau window. On trigger, refine to single-step
    resolution using suffix/prefix maxima over the two adjacent windows.
    """
    buf   = buf_init.copy()
    theta = buf[-1]
    win   = round(win_tau * TAU / DT)
    t_max = int(t_max_tau * TAU / DT)
    prev_win = np.full(win, theta)   # θ over previous win_tau window
    curr_win = np.empty(win)

    for step in range(t_max):
        k1    = _f(buf[0], k)
        k2    = _f(buf[1], k)
        theta = theta + 0.5 * (k1 + k2) * DT
        buf[:-1] = buf[1:]
        buf[-1]  = theta
        curr_win[step % win] = theta

        if (step + 1) % win == 0:
            if curr_win.max() < np.pi:
                # Refine: find first endpoint j in [1, win] s.t.
                #   max(prev_win[j:], curr_win[:j]) < π
                # using O(win) suffix/prefix max arrays.
                t_win_start   = (step + 1 - win) * DT
                suffix_max    = np.maximum.accumulate(prev_win[::-1])[::-1]
                prefix_max    = np.maximum.accumulate(curr_win)
                combined      = np.empty(win)
                combined[:-1] = np.maximum(suffix_max[1:], prefix_max[:-1])
                combined[-1]  = prefix_max[-1]
                first_j = int(np.argmax(combined < np.pi))
                return (t_win_start + (first_j + 1) * DT) / TAU
            prev_win = curr_win.copy()
            curr_win = np.empty(win)
    return t_max_tau


def main():
    tauk_ic  = 4.13
    tauk_c   = 3.9849  # onset between 3.9848 and 3.9849
    t_warmup = 100_000.0
    t_max    = 10_000.0

    amp_scan = np.concatenate([
        np.linspace(3.92, 3.984, 6),     # below onset → 4τ LC
        np.linspace(3.9849, 4.04, 9),    # above onset → LC2
    ])
    esc_scan = np.concatenate([
        np.linspace(3.80, 3.96, 9),
        np.linspace(3.965, 3.982, 10),
        np.array([3.9830, 3.9835, 3.9838, 3.9840, 3.9842,
                  3.9844, 3.9845, 3.9846, 3.9847, 3.98472,
                  3.98474, 3.98476, 3.98478, 3.9848, 3.98481, 
                  3.98482]),
    ])

    print(f"Spinning up reference at τk = {tauk_ic} for t = {t_warmup:.0f} …")
    buf_ref = spinup_dde(tauk_ic / TAU, 2.0, t_warmup)
    print(f"  θ(end) = {buf_ref[-1]:.4f}")

    print(f"\nAmplitude / period scan ({len(amp_scan)} τk values) …")
    amps, pers = [], []
    for tauk in amp_scan:
        buf = spinup_dde(tauk / TAU, buf_ref, t_warmup=5_000.0)
        a, p = measure_amplitude_period(tauk / TAU, buf)
        amps.append(a); pers.append(p)
        print(f"  τk = {tauk:.4f}  R = {a:.4f}  T/τ = {p/TAU:.3f}")
    amps, pers = np.array(amps), np.array(pers)

    print(f"\nEscape time scan ({len(esc_scan)} τk values) …")
    t_esc = []
    for tauk in esc_scan:
        t = escape_time_tau(tauk / TAU, buf_ref, t_max_tau=t_max)
        flag = "  (censored)" if t >= t_max else ""
        print(f"  τk = {tauk:.5f}  T_esc = {t:.1f} τ{flag}")
        t_esc.append(t)
    t_esc = np.array(t_esc)

    # Plot
    fig = plt.figure(figsize=(14, 13))
    gs  = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.35)
    ax_amp = fig.add_subplot(gs[0, 0])
    ax_per = fig.add_subplot(gs[0, 1])
    ax_esc = fig.add_subplot(gs[1, :])

    above = amp_scan >= tauk_c

    # (a) amplitude
    ax_amp.scatter(amp_scan[~above], amps[~above], color="gray", s=30, zorder=3, label="below onset → LC1")
    ax_amp.scatter(amp_scan[above],  amps[above],  color="#2166AC", s=30, zorder=3, label="above onset → LC2")
    ax_amp.axhline(np.pi, color="#D6604D", lw=1.2, ls="--", alpha=0.8, label=r"$R = \pi$")
    ax_amp.axvline(tauk_c, color="red", lw=1.0, ls=":", alpha=0.7, label=rf"$(\tau k)_c = {tauk_c}$")
    ax_amp.set_xlabel(r"$\tau k$", fontsize=11)
    ax_amp.set_ylabel(r"Half-amplitude $R$", fontsize=11)
    ax_amp.set_title("(a)  Amplitude near onset\n"
                     r"$R \to \pi$ at onset  (Hopf would give $R \to 0$)", fontsize=10)
    ax_amp.legend(fontsize=8); ax_amp.grid(True, linewidth=0.4, alpha=0.5)

    # (b) period
    valid  = above & np.isfinite(pers)
    T_an   = (4.0 * TAU + 4.0 * np.sqrt(8.0 * TAU**2 * np.maximum(amps[valid] - np.pi, 0) / np.pi**3)) / TAU
    ax_per.scatter(amp_scan[valid], pers[valid] / TAU, color="#2166AC", s=30, zorder=3, label="measured LC2")
    ax_per.plot(amp_scan[valid], T_an, color="#D6604D", lw=1.8, ls="--", zorder=4,
                label=r"$4\tau + 4\sqrt{8\tau^2(R-\pi)/\pi^3}$")
    ax_per.axvline(tauk_c, color="red", lw=1.0, ls=":", alpha=0.7)
    ax_per.set_xlabel(r"$\tau k$", fontsize=11)
    ax_per.set_ylabel(r"Period $T\,/\,\tau$", fontsize=11)
    ax_per.set_title("(b)  Period near onset\n"
                     r"$T \to 4\tau$ at onset  (SNIC would give $T \to \infty$)", fontsize=10)
    ax_per.legend(fontsize=8); ax_per.grid(True, linewidth=0.4, alpha=0.5)

    # (c) escape time
    delta   = tauk_c - esc_scan
    escaped = t_esc < t_max
    ax_esc.scatter(delta[escaped],  t_esc[escaped], color="#2166AC", s=40, zorder=3,
                   label=r"$T_{esc}$ (escaped to 4$\tau$ LC)")
    ax_esc.scatter(delta[~escaped], np.full((~escaped).sum(), t_max * 0.85),
                   color="#2166AC", s=40, marker="v", alpha=0.45, zorder=3,
                   label=rf"censored ($T_{{esc}} > {t_max:.0f}\tau$)")
    if escaped.sum() >= 3:
        log_C  = float(np.mean(np.log(t_esc[escaped]) + 0.5 * np.log(delta[escaped])))
        C_fit  = np.exp(log_C)
        d_line = np.geomspace(delta[escaped].min() * 0.4, delta[escaped].max() * 2.0, 300)
        ax_esc.plot(d_line, C_fit * d_line**(-0.5), color="#D6604D", lw=2.0, ls="--",
                    label=rf"saddle-node fit: $C/\sqrt{{|\Delta\tau k|}}$,  $C={C_fit:.1f}\tau$")
        slope, _ = np.polyfit(np.log(delta[escaped]), np.log(t_esc[escaped]), 1)
        ax_esc.annotate(rf"measured slope: ${slope:.2f}$  (theory: $-0.5$)",
                        xy=(0.04, 0.10), xycoords="axes fraction", fontsize=10, color="#2166AC",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2166AC", alpha=0.8))
    ax_esc.set_xscale("log"); ax_esc.set_yscale("log")
    ax_esc.set_xlabel(rf"$|\tau k - (\tau k)_c|$,  $(\tau k)_c = {tauk_c}$", fontsize=11)
    ax_esc.set_ylabel(r"$T_{esc}\,/\,\tau$", fontsize=11)
    ax_esc.set_title("(c)  Ghost / bottleneck escape time below LC2 onset\n"
                     r"Saddle-node: $T_{esc} \propto |\Delta(\tau k)|^{-1/2}$  (slope $= -1/2$ on log-log)"
                     "\n"
                     r"Homoclinic would give $T_{esc} \propto -\ln|\Delta|$ instead", fontsize=10)
    ax_esc.legend(fontsize=9, loc="upper right")
    ax_esc.grid(True, which="both", linewidth=0.4, alpha=0.5)

    fig.suptitle(rf"Saddle-node bifurcation at LC2 onset  ($\tau=25$, ref IC at $\tau k={tauk_ic}$)"
                 "\n" r"$\mathrm{d}\theta/\mathrm{d}t = -k\sin(\theta(t-\tau))$", fontsize=13)

    out = os.path.join(SCRIPT_DIR, "plots", "transients", "lc2_bifurcation_type.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
