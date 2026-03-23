import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(SCRIPT_DIR, "outputs", "bifurcation4IC_zoom")
PLOT_ROOT = os.path.join(SCRIPT_DIR, "plots")
PLOT_DIRS = {
    "lc2": os.path.join(PLOT_ROOT, "lc2"),
    "intermittency": os.path.join(PLOT_ROOT, "intermittency"),
}

BRANCH_META: Dict[str, Dict[str, str]] = {
    "p4_pos": {"label": r"$\theta_0=+1$", "color": "#901A1E"},
    "p4_neg": {"label": r"$\theta_0=-1$", "color": "#901A1E"},
    "lc2_pos": {"label": r"$\theta_0=+2$", "color": "#901A1E"},
    "lc2_neg": {"label": r"$\theta_0=-2$", "color": "#901A1E"},
}

PANELS = [
    ("lc2", (4.100, 4.250), "LC2 Zoom"),
    ("intermittency", (4.800, 5.000), "Intermittency Zoom"),
]

TWO_PI = 2.0 * np.pi


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 28,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 14,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.3, linewidth=0.7)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=1)


def parse_branch(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"(p4_pos|p4_neg|lc2_pos|lc2_neg)_", base)
    return m.group(1) if m else "unknown"


def load_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            _header = f.readline()
            first_data = f.readline()
            if first_data.strip() == "":
                return np.array([]), np.array([])
    except OSError:
        return np.array([]), np.array([])

    try:
        arr = np.loadtxt(path, delimiter="\t", skiprows=1)
    except Exception:
        return np.array([]), np.array([])

    if arr.ndim == 1 and arr.size == 5:
        arr = arr.reshape(1, 5)
    if arr.size == 0:
        return np.array([]), np.array([])

    tauk_real = arr[:, 0]
    theta_max = np.mod(arr[:, 4], TWO_PI)
    finite = np.isfinite(tauk_real) & np.isfinite(theta_max)
    return tauk_real[finite], theta_max[finite]


def plot_panel(
    ax: plt.Axes,
    folder: str,
    xlim: Tuple[float, float],
    title: str,
    y_limits: Tuple[float, float] | None = None,
    use_pi_ticks: bool = True,
) -> None:
    files = sorted(glob.glob(os.path.join(OUT_ROOT, folder, "*.tsv")))
    if not files:
        ax.text(0.5, 0.5, f"No data in {folder}", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(*xlim)
        if y_limits is None:
            ax.set_ylim(0.0, TWO_PI)
        else:
            ax.set_ylim(*y_limits)
        style_axes(ax)
        ax.set_title(title)
        return

    grouped: Dict[str, List[str]] = {k: [] for k in BRANCH_META}
    for fp in files:
        branch = parse_branch(fp)
        if branch in grouped:
            grouped[branch].append(fp)

    for branch, paths in grouped.items():
        xs: List[float] = []
        ys: List[float] = []
        for p in paths:
            x, y = load_points(p)
            if x.size == 0:
                continue
            xs.extend(x.tolist())
            ys.extend(y.tolist())
        if not xs:
            continue

        meta = BRANCH_META[branch]
        ax.plot(
            np.array(xs),
            np.array(ys),
            linestyle="None",
            marker=".",
            markersize=0.1,
            color=meta["color"],
            alpha=0.50,
            label=meta["label"],
        )

    ax.set_xlim(*xlim)
    if y_limits is None:
        ax.set_ylim(0.0, TWO_PI)
    else:
        ax.set_ylim(*y_limits)
    if use_pi_ticks and y_limits is None:
        ax.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, TWO_PI])
        ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_title(title)
    style_axes(ax)


def main() -> None:
    set_plot_style()

    saved: List[str] = []
    for folder, xlim, title in PANELS:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        fig_single.patch.set_alpha(0.0)
        panel_ylim = (4.6, 5.3) if folder == "lc2" else None
        plot_panel(ax_single, folder, xlim, title, y_limits=panel_ylim)
        ax_single.set_xlabel(r"$k\tau$")
        ax_single.set_ylabel(r"$\theta_{\max}$")

        out_dir = PLOT_DIRS[folder]
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"bifurcation_4ic_{folder}_zoom")
        fig_single.savefig(base + ".png", format="png", transparent=True, facecolor="none", edgecolor="none")
        plt.close(fig_single)
        saved.append(base + ".png")

        if folder == "intermittency":
            fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
            fig_zoom.patch.set_alpha(0.0)
            plot_panel(
                ax_zoom,
                folder,
                xlim,
                "Intermittency Zoom (y: 2.4-3.2)",
                y_limits=(2.6, 3.1),
                use_pi_ticks=False,
            )
            ax_zoom.set_xlabel(r"$k\tau$")
            ax_zoom.set_ylabel(r"$\theta_{\max}$")

            base_zoom = os.path.join(out_dir, "bifurcation_4ic_intermittency_zoom_y2p4_3p2")
            fig_zoom.savefig(base_zoom + ".png", format="png", transparent=True, facecolor="none", edgecolor="none")
            plt.close(fig_zoom)
            saved.append(base_zoom + ".png")

    print("Saved:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
