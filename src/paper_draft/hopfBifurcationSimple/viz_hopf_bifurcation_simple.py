import math
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "plots")

# DDE parameters
TAU = 1.0
THETA0 = math.pi / 8.0
DT = 0.005
T_MAX = 40.0
GROWTH_SPEED = 1.0

# Hopf threshold: k * tau = pi/2
K_HOPF = math.pi / (2.0 * TAU)

# Three regimes around Hopf bifurcation.
CASES = [
    {
        "label": "below Hopf",
        "k": 0.4,
        "color": "#547AA5",
    },
    {
        "label": "near Hopf",
        "k": 1.4,
        "color": "#CBA810",
    },
    {
        "label": "above Hopf",
        "k": 1.6,
        "color": "#901A1E",
    },
]

HOPF_LINE_COLOR = "#666666"


def delayed_theta(theta_series, idx, delay_steps):
    """Return theta(t-tau) using nearest-step delay; prehistory is constant THETA0."""
    delayed_idx = idx - delay_steps
    if delayed_idx < 0:
        return THETA0
    return theta_series[delayed_idx]


def simulate_dde(tau, k, theta0, dt, t_max):
    """Simulate dtheta/dt = -k sin(theta(t-tau)) with explicit Euler."""
    n_steps = int(round(t_max / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    theta = np.empty(n_steps + 1, dtype=float)
    theta[0] = theta0

    delay_steps = max(1, int(round(tau / dt)))
    for i in range(n_steps):
        theta_tau = delayed_theta(theta, i, delay_steps)
        dtheta = -k * math.sin(theta_tau)
        theta[i + 1] = theta[i] + dt * dtheta

    return t, theta


def build_root_path(theta, dt, speed):
    """Integrate a planar path with heading angle theta measured from vertical."""
    x = np.empty_like(theta)
    y = np.empty_like(theta)
    x[0] = 0.0
    y[0] = 0.0

    for i in range(len(theta) - 1):
        x[i + 1] = x[i] + speed * math.sin(theta[i]) * dt
        y[i + 1] = y[i] - speed * math.cos(theta[i]) * dt

    return x, y


def style_main_axes(ax):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.3)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=1)


def main():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 30,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    fig = plt.figure(figsize=(15.0, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.5])
    fig.set_constrained_layout_pads(wspace=0.08, hspace=0.05)
    fig.patch.set_alpha(0.0)

    # Shrink the total vertical span of the left stack a bit for better balance.
    left_gs = gs[0, 0].subgridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.001])
    ax_a1 = fig.add_subplot(left_gs[0, 0])
    ax_a2 = fig.add_subplot(left_gs[1, 0], sharex=ax_a1)
    ax_a3 = fig.add_subplot(left_gs[2, 0], sharex=ax_a1)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_ts_list = [ax_a1, ax_a2, ax_a3]

    all_root_x = []
    all_root_y = []
    root_paths = []
    root_start_points = []
    theta_series = []

    for row, case in enumerate(CASES):
        k = case["k"]
        color = case["color"]
        label = case["label"]

        t, theta = simulate_dde(TAU, k, THETA0, DT, T_MAX)
        x_root, y_root = build_root_path(theta, DT, GROWTH_SPEED)

        theta_series.append(theta)
        all_root_x.append(x_root)
        all_root_y.append(y_root)
        root_paths.append((x_root, y_root, color, label, k))

        ax_ts = ax_ts_list[row]

        ax_ts.plot(t, theta, color=color, lw=2.0)
        ax_ts.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        if row == len(CASES) - 1:
            ax_ts.set_xlabel(r"$t$")
        ax_ts.set_ylabel(r"$\theta(t)$")
        ax_ts.set_title(rf"$k \tau={k:.2f}$ ({label})", pad=8)
        style_main_axes(ax_ts)

    # Hide repeated x tick labels on upper time-series axes.
    for ax in (ax_a1, ax_a2):
        ax.tick_params(labelbottom=False)

    # Use identical y-axis limits/ticks and fixed label coordinates across A1/A2/A3.
    ts_ymin = min(float(np.min(v)) for v in theta_series)
    ts_ymax = max(float(np.max(v)) for v in theta_series)
    ts_ypad = 0.08 * max(1e-8, ts_ymax - ts_ymin)
    for ax in ax_ts_list:
        ax.set_ylim(ts_ymin - ts_ypad, ts_ymax + ts_ypad)
        ax.yaxis.set_label_coords(-0.11, 0.5)

    # Single root-space panel with slight x offsets so trajectories are visually separated.
    x_span = 50*max(float(np.max(v)) - float(np.min(v)) for v in all_root_x)
    x_offsets = np.linspace(-0.25 * x_span, 0.25 * x_span, len(root_paths))
    for offset, (x_root, y_root, color, label, k) in zip(x_offsets, root_paths):
        x_shifted = x_root + offset
        ax_b.plot(x_shifted, y_root, color=color, lw=2.2, label=rf"$k \tau={k:.2f}$ ({label})")
        ax_b.scatter([x_shifted[0]], [y_root[0]], s=22, color="black", zorder=4)
        root_start_points.append((float(x_shifted[0]), float(y_root[0])))

    # Keep comparable spatial scales across root panels.
    x_min = min(float(np.min(v)) for v in all_root_x)
    x_max = max(float(np.max(v)) for v in all_root_x)
    y_min = min(float(np.min(v)) for v in all_root_y)
    y_max = max(float(np.max(v)) for v in all_root_y)

    x_pad = 2 * max(1e-8, x_max - x_min)
    y_pad = 0.04 * max(1e-8, y_max - y_min)

    ax_b.set_xlim((x_min + np.min(x_offsets)) - x_pad, (x_max + np.max(x_offsets)) + x_pad)
    ax_b.set_ylim(y_min - y_pad, y_max + y_pad)

    # # Vertical guide lines dropping from each root's initial point.
    # y_bottom = y_min - y_pad
    # for x0, y0 in root_start_points:
    #     ax_b.plot([x0, x0], [y0, y_bottom], color=HOPF_LINE_COLOR, lw=0.8, alpha=0.45, zorder=1)

    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xticklabels([])  # hide x tick labels since they are not meaningful here
    ax_b.set_yticklabels([])  # hide y tick labels since they are not meaningful here
    ax_b.set_title("Root tip trajectory")
    style_main_axes(ax_b)
    ax_b.grid(False)


    os.makedirs(OUT_DIR, exist_ok=True)
    out_svg = os.path.join(OUT_DIR, "hopf_bifurcation_simple.svg")
    out_pdf = os.path.join(OUT_DIR, "hopf_bifurcation_simple.pdf")
    fig.savefig(out_svg, format="svg", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(out_pdf, format="pdf", transparent=True, facecolor="none", edgecolor="none")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
