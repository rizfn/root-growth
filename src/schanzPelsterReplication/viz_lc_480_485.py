"""Run C++ target-IC timeseries sims and plot timeseries + phase portraits.

This script compiles and runs timeseries_target_ic.cpp for tau*k targets 4.80
and 4.85 across ICs [+1, -1, +2, -2], writes TSV output to a dedicated
outputs directory, and then generates the visualization.
"""

import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TAU = 25.0
DT = 0.01
RECORD_DT = 0.05
T_WARMUP_TARGET = 3000.0
T_WARMUP_REAL = 12000.0
T_MEASURE = 2000.0

ICS_SHOW = [1.0, -1.0, 2.0, -2.0]
TARGET_TAUK = [4.80, 4.85, 4.90]

LC2_TARGET_APPEAR = 4.13
LC2_TARGET_DISAPPEAR = 4.24
LC2_APPEAR_CUTOFF = 3.985
LC2_DISAPPEAR_CUTOFF = 4.24

IC_PALETTE = {
    1.0: "#2166AC",
    -1.0: "#92C5DE",
    2.0: "#D6604D",
    -2.0: "#F4A582",
}


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Map angle values to the principal interval (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def align_to_upward_zero(theta: np.ndarray, t: np.ndarray, window: float):
    """Select a phase-anchored window using the last upward crossing before end-window."""
    if theta is None or t is None or len(theta) < 3 or len(t) < 3:
        return np.array([]), np.array([])

    t_end = float(t[-1])
    t_ref = t_end - window

    last_cross = None
    for i in range(len(theta) - 1):
        y0 = float(theta[i])
        y1 = float(theta[i + 1])
        if not (y0 < 0.0 <= y1):
            continue
        t0 = float(t[i])
        t1 = float(t[i + 1])
        dy = y1 - y0
        if abs(dy) < 1e-15:
            t_cross = t0
        else:
            frac = -y0 / dy
            t_cross = t0 + frac * (t1 - t0)
        if t_cross <= t_ref:
            last_cross = t_cross

    if last_cross is None:
        t_start = max(float(t[0]), t_ref)
    else:
        t_start = last_cross

    t_stop = min(t_end, t_start + window)
    mask = (t >= t_start) & (t <= t_stop)
    if np.sum(mask) < 3:
        return np.array([]), np.array([])

    theta_sel = theta[mask]
    t_rel = t[mask] - t_start
    return theta_sel, t_rel


def target_tauk_for_ic(tauk_real: float, ic: float) -> float:
    """Match target-IC policy used in the zoom workflow for branch tracking."""
    if abs(ic) < 1.5:
        return tauk_real
    if tauk_real < LC2_APPEAR_CUTOFF:
        return LC2_TARGET_APPEAR
    if tauk_real > LC2_DISAPPEAR_CUTOFF:
        return LC2_TARGET_DISAPPEAR
    return tauk_real


def ic_tag(ic: float) -> str:
    s = f"{ic:.1f}"
    return s.replace("-", "n")


def run_cmd(cmd, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def compile_solver(script_dir: Path, exe_path: Path) -> None:
    cpp_path = script_dir / "timeseries_target_ic.cpp"
    cmd = [
        "g++",
        "-fdiagnostics-color=always",
        "-std=c++17",
        "-O2",
        "-o",
        str(exe_path),
        str(cpp_path),
    ]
    print("Compiling timeseries_target_ic.cpp ...")
    run_cmd(cmd, script_dir)


def simulate_all(script_dir: Path):
    exe_path = script_dir / "timeseries_target_ic"
    out_dir = script_dir / "outputs" / "timeseries_480_485"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = {}
    pending = []
    for tauk_real in TARGET_TAUK:
        generated[tauk_real] = {}
        for ic in ICS_SHOW:
            tauk_target = target_tauk_for_ic(tauk_real, ic)

            out_name = (
                f"tauk_{tauk_real:.6f}_target_{tauk_target:.6f}"
                f"_ic_{ic_tag(ic)}.tsv"
            )
            out_path = out_dir / out_name
            generated[tauk_real][ic] = out_path

            if out_path.exists():
                print(
                    f"  using existing tau*k={tauk_real:.3f}, ic={ic:+.1f}, "
                    f"target={tauk_target:.3f}"
                )
                continue

            pending.append((tauk_real, tauk_target, ic, out_path))

    if pending:
        compile_solver(script_dir, exe_path)
        for tauk_real, tauk_target, ic, out_path in pending:
            k_real = tauk_real / TAU
            k_target = tauk_target / TAU
            cmd = [
                str(exe_path),
                f"{TAU}",
                f"{k_real:.10f}",
                f"{k_target:.10f}",
                f"{ic:.6f}",
                f"{DT}",
                f"{T_WARMUP_TARGET}",
                f"{T_WARMUP_REAL}",
                f"{T_MEASURE}",
                f"{RECORD_DT}",
                str(out_path),
            ]
            run_cmd(cmd, script_dir)
            print(
                f"  simulated tau*k={tauk_real:.3f}, ic={ic:+.1f}, "
                f"target={tauk_target:.3f}"
            )

    return generated


def load_ts(path: Path):
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        return None, None
    return data[:, 0], data[:, 1]


def make_plot(script_dir: Path, generated):
    plot_dir = script_dir / "plots" / "bifurcation_480_485"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(TARGET_TAUK),
        2,
        figsize=(13, 4.4 * len(TARGET_TAUK)),
        gridspec_kw={"wspace": 0.15, "hspace": 0.30},
    )

    if len(TARGET_TAUK) == 1:
        axes = np.array([axes])

    for row_i, tauk_real in enumerate(TARGET_TAUK):
        ax_ts, ax_phase = axes[row_i, 0], axes[row_i, 1]
        k_real = tauk_real / TAU

        for ic in ICS_SHOW:
            path = generated[tauk_real][ic]
            t, theta = load_ts(path)
            if t is None or len(t) < 5:
                continue

            window = 4 * 4 * TAU
            theta_wrapped = wrap_to_pi(theta)
            th_ts_aligned, x_ts = align_to_upward_zero(theta_wrapped, t, window)
            if th_ts_aligned.size < 3 or x_ts.size < 3:
                continue
            color = IC_PALETTE.get(ic, "gray")

            ax_ts.plot(
                x_ts,
                th_ts_aligned,
                lw=1.1,
                color=color,
                label=f"IC = {ic:+.1f}",
            )

            dtheta = np.gradient(th_ts_aligned, x_ts)
            ax_phase.plot(
                th_ts_aligned,
                dtheta,
                lw=0.8,
                color=color,
                alpha=0.8,
                label=f"IC = {ic:+.1f}",
            )

        ax_ts.set_title(f"tau*k = {tauk_real:.4f} (k = {k_real:.5f})", fontsize=9)
        ax_ts.set_xlabel("aligned time (8 periods)")
        ax_ts.set_ylabel("theta(t)")
        ax_ts.set_ylim(-np.pi, np.pi)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(fontsize=8)

        ax_phase.set_xlabel("theta")
        ax_phase.set_ylabel("dtheta/dt")
        ax_phase.grid(True, alpha=0.3)
        ax_phase.legend(fontsize=8)

    fig.suptitle(
        "DDE: dtheta/dt = -k*sin(theta(t-tau)); C++ target-IC sims at tau*k=4.80, 4.85",
        fontsize=11,
    )

    out = plot_dir / "lc_480_485.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    script_dir = Path(__file__).resolve().parent
    generated = simulate_all(script_dir)
    make_plot(script_dir, generated)


if __name__ == "__main__":
    main()
