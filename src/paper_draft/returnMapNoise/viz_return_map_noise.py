import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "plots"

# Default simulation parameters used to locate the source folder.
TAU = 1.0
K = 3.0
THETA0 = 1.5708
DT = 0.01
TMAX = 4000.0
ITERATE_N = 1
TRANSIENT_FRAC = 0.5
MAX_POINTS_PER_ETA = 7000
SEED = 53

DIAG_COLOR = "#666666"
POINT_COLOR = "#901A1E"


def folder_name(tau, k, theta0, dt, tmax):
    return f"tau_{tau:g}_k_{k:g}_theta0_{theta0:g}_dt_{dt:g}_tmax_{tmax:g}"


def candidate_source_dirs(tau, k, theta0, dt, tmax):
    name = folder_name(tau, k, theta0, dt, tmax)
    return [
        SCRIPT_DIR / "outputs" / "SDDETimeseries" / name,
        SCRIPT_DIR / "outputs" / "SDDETimeseries" / "tau_k_raster" / name,
        SCRIPT_DIR / "outputs" / "SDDETimeseries" / "long" / name,
        SCRIPT_DIR.parent.parent / "delayDETimeseries" / "outputs" / "SDDETimeseries" / name,
        SCRIPT_DIR.parent.parent / "delayDETimeseries" / "outputs" / "SDDETimeseries" / "tau_k_raster" / name,
        SCRIPT_DIR.parent.parent / "delayDETimeseries" / "outputs" / "SDDETimeseries" / "long" / name,
    ]


def source_roots():
    return [
        SCRIPT_DIR / "outputs" / "SDDETimeseries",
        SCRIPT_DIR / "outputs" / "SDDETimeseries" / "tau_k_raster",
        SCRIPT_DIR / "outputs" / "SDDETimeseries" / "long",
        SCRIPT_DIR.parent.parent / "delayDETimeseries" / "outputs" / "SDDETimeseries",
        SCRIPT_DIR.parent.parent / "delayDETimeseries" / "outputs" / "SDDETimeseries" / "tau_k_raster",
        SCRIPT_DIR.parent.parent / "delayDETimeseries" / "outputs" / "SDDETimeseries" / "long",
    ]


def parse_folder_params(folder_name_str):
    pattern = (
        r"^tau_(?P<tau>[-+0-9.eE]+)_k_(?P<k>[-+0-9.eE]+)_theta0_(?P<theta0>[-+0-9.eE]+)"
        r"_dt_(?P<dt>[-+0-9.eE]+)_tmax_(?P<tmax>[-+0-9.eE]+)$"
    )
    m = re.match(pattern, folder_name_str)
    if not m:
        return None
    try:
        return {
            "tau": float(m.group("tau")),
            "k": float(m.group("k")),
            "theta0": float(m.group("theta0")),
            "dt": float(m.group("dt")),
            "tmax": float(m.group("tmax")),
        }
    except ValueError:
        return None


def approx_equal(a, b, rel=1e-9, abs_tol=1e-9):
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1.0))


def fallback_scan_source_folder(tau, k, theta0, dt, tmax):
    candidates = []
    for root in source_roots():
        if not root.exists():
            continue
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            params = parse_folder_params(folder.name)
            if params is None:
                continue

            if not approx_equal(params["tau"], tau, rel=1e-6):
                continue
            if not approx_equal(params["k"], k, rel=1e-6):
                continue

            score = (
                abs(params["theta0"] - theta0)
                + abs(params["dt"] - dt)
                + abs(params["tmax"] - tmax)
            )
            candidates.append((score, folder))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_source_folder(tau, k, theta0, dt, tmax):
    for path in candidate_source_dirs(tau, k, theta0, dt, tmax):
        if path.exists():
            return path
    return fallback_scan_source_folder(tau, k, theta0, dt, tmax)


def parse_eta(path):
    match = re.search(r"eta_([^_]+)_simNo_", str(path))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def discover_etas(folder):
    files = sorted(folder.glob("eta_*_simNo_*.tsv"))
    eta_values = sorted({parse_eta(f) for f in files if parse_eta(f) is not None})
    return eta_values


def create_return_map(theta, tau_steps, n):
    if tau_steps <= 0 or n <= 0:
        return np.array([]), np.array([])

    jump = n * tau_steps
    max_start = len(theta) - jump
    if max_start <= 0:
        return np.array([]), np.array([])

    idx = np.arange(0, max_start, tau_steps)
    x = theta[idx]
    y = theta[idx + jump]
    return x, y


def load_return_points(file_path, tau, dt, n, transient_frac):
    try:
        data = np.loadtxt(file_path, skiprows=1, usecols=(1,))
    except Exception:
        return np.array([]), np.array([])

    if data.ndim == 0:
        data = np.array([float(data)])
    if len(data) < 10:
        return np.array([]), np.array([])

    transient_idx = int(transient_frac * len(data))
    theta = data[transient_idx:]
    if len(theta) < 10:
        return np.array([]), np.array([])

    tau_steps = max(1, int(round(tau / dt)))
    x, y = create_return_map(theta, tau_steps, n)
    if len(x) == 0:
        return x, y

    two_pi = 2.0 * math.pi
    # Wrap angles to [-pi, pi) for symmetric phase-space view.
    x_wrapped = (x + math.pi) % two_pi - math.pi
    y_wrapped = (y + math.pi) % two_pi - math.pi
    return x_wrapped, y_wrapped


def gather_points_for_eta(folder, eta, tau, dt, n, transient_frac):
    eta_str = "0" if eta == 0.0 else str(eta)
    files = sorted(folder.glob(f"eta_{eta_str}_simNo_*.tsv"))

    xs = []
    ys = []
    for file_path in files:
        x, y = load_return_points(file_path, tau, dt, n, transient_frac)
        if len(x) > 0:
            xs.append(x)
            ys.append(y)

    if not xs:
        return np.array([]), np.array([]), 0

    return np.concatenate(xs), np.concatenate(ys), len(files)


def pick_grid(n_panels):
    return 1, n_panels


def format_eta(eta):
    if eta == 0.0:
        return "0"
    return f"{eta:g}"


def style_axes(ax):
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([-math.pi, math.pi])
    ax.set_yticks([-math.pi, math.pi])
    ax.set_xticklabels([r"$-\pi$", r"$\pi$"])
    ax.set_yticklabels([r"$-\pi$", r"$\pi$"])

    ax.minorticks_on()
    ax.grid(True, which="both", alpha=0.28)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=1)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)


def plot_return_maps(tau, k, theta0, dt, tmax, n):
    src = find_source_folder(tau, k, theta0, dt, tmax)
    if src is None:
        print("Could not find source outputs folder for parameter set.")
        for cand in candidate_source_dirs(tau, k, theta0, dt, tmax):
            print(f"  tried: {cand}")
        return

    eta_values = discover_etas(src)
    if not eta_values:
        print(f"No eta_* files found in: {src}")
        return

    print(f"Using source folder: {src}")
    print(f"Found {len(eta_values)} noise values: {[format_eta(v) for v in eta_values]}")

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 24,
            "axes.titlesize": 22,
            "axes.labelsize": 30,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    nrows, ncols = pick_grid(len(eta_values))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 5.0 * nrows), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.08, hspace=0.06)
    fig.patch.set_alpha(0.0)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    rng = np.random.default_rng(SEED)
    two_pi = 2.0 * math.pi

    for i, eta in enumerate(eta_values):
        ax = axes[i]
        x, y, n_sims = gather_points_for_eta(src, eta, tau, dt, n, TRANSIENT_FRAC)

        if len(x) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            style_axes(ax)
            continue

        if len(x) > MAX_POINTS_PER_ETA:
            idx = rng.choice(len(x), size=MAX_POINTS_PER_ETA, replace=False)
            x = x[idx]
            y = y[idx]

        ax.scatter(x, y, s=10, alpha=0.34, color=POINT_COLOR, edgecolors="none")
        ax.plot(
            [-math.pi, math.pi],
            [-math.pi, math.pi],
            ls=(0, (8, 6)),
            lw=1.0,
            color=DIAG_COLOR,
            alpha=0.75,
        )

        style_axes(ax)
        ax.set_xlabel(r"$\theta(m\tau)$", labelpad=-14)
        ax.set_ylabel(r"$\theta(m\tau + n\tau)$", labelpad=-22)
        ax.text(
            0.03,
            0.95,
            rf"$\eta={format_eta(eta)}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=26,
        )

    for j in range(len(eta_values), len(axes)):
        axes[j].set_visible(False)

    os.makedirs(OUT_DIR, exist_ok=True)
    iterate_suffix = "" if n == 1 else f"_{n}th_iterate"
    base_name = f"return_maps_noise_tau_{tau:g}_k_{k:g}{iterate_suffix}"
    out_svg = OUT_DIR / f"{base_name}.svg"
    out_pdf = OUT_DIR / f"{base_name}.pdf"

    fig.savefig(out_svg, format="svg", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(out_pdf, format="pdf", transparent=True, facecolor="none", edgecolor="none")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_pdf}")


def main():
    tau = TAU
    k = K
    theta0 = THETA0
    dt = DT
    tmax = TMAX
    n = ITERATE_N

    if len(sys.argv) > 1:
        tau = float(sys.argv[1])
    if len(sys.argv) > 2:
        k = float(sys.argv[2])
    if len(sys.argv) > 3:
        theta0 = float(sys.argv[3])
    if len(sys.argv) > 4:
        dt = float(sys.argv[4])
    if len(sys.argv) > 5:
        tmax = float(sys.argv[5])
    if len(sys.argv) > 6:
        n = int(sys.argv[6])

    print(
        "Generating return maps with params: "
        f"tau={tau:g}, k={k:g}, theta0={theta0:g}, dt={dt:g}, tmax={tmax:g}, n={n}"
    )
    plot_return_maps(tau, k, theta0, dt, tmax, n)


if __name__ == "__main__":
    main()
