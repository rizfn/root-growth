import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DATA_DIR = os.path.join(SCRIPT_DIR, "outputs", "bifurcation4IC_target")
ZOOM_DATA_ROOT = os.path.join(SCRIPT_DIR, "outputs", "bifurcation4IC_zoom")
PLOT_ROOT = os.path.join(SCRIPT_DIR, "plots")
PLOT_DIR = os.path.join(PLOT_ROOT, "total")

BRANCH_META: Dict[str, Dict[str, str]] = {
    "p4_pos": {"label": r"$\theta_0=+1$", "color": "#901A1E"},
    "p4_neg": {"label": r"$\theta_0=-1$", "color": "#901A1E"},
    "lc2_pos": {"label": r"$\theta_0=+2$", "color": "#901A1E"},
    "lc2_neg": {"label": r"$\theta_0=-2$", "color": "#901A1E"},
}

HOPF_COLOR = "#666666"
TWO_PI = 2.0 * np.pi
INSET_FRAME_LW = 0.9
OUTER_BORDER_LW = 1.2

YELLOW_COLOR = "#CBA810"
ZOOM_BOX_LW = 2.0
ZOOM_OUTER_BORDER_LW = 2.0

INSET_SPECS = [
    {
        "folder": "lc2",
        "xlim": (4.100, 4.250),
        "ylim": (4.6, 5.3),
        "bounds": [0.43, 0.57, 0.30, 0.35],
        "color": "#547AA5",
    },
    {
        "folder": "intermittency",
        "xlim": (4.800, 5.000),
        "ylim": (2.6, 3.1),
        "bounds": [0.08, 0.57, 0.30, 0.35],
        "color": YELLOW_COLOR,
    },
]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 17,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def style_axes(
    ax: plt.Axes,
    *,
    show_tick_labels: bool = True,
    facecolor: str = "none",
    show_grid: bool = True,
) -> None:
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(INSET_FRAME_LW)
    ax.minorticks_off()
    ax.grid(show_grid, which="major", alpha=0.3, linewidth=0.7)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=8, length=0)
    if not show_tick_labels:
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
        )


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


def collect_grouped_files(files: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {k: [] for k in BRANCH_META}
    for fp in files:
        branch = parse_branch(fp)
        if branch in grouped:
            grouped[branch].append(fp)
    return grouped


def plot_grouped_cloud(ax: plt.Axes, grouped: Dict[str, List[str]], *, add_labels: bool) -> None:
    for branch, paths in grouped.items():
        if not paths:
            continue

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
            marker="o",
            markersize=0.5,
            markeredgewidth=0,
            color=meta["color"],
            alpha=0.50,
            label=meta["label"] if add_labels else None,
        )


def draw_zoom_box(ax: plt.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float], color: str) -> None:
    # Ensure the inner edge of the stroked rectangle aligns with xlim/ylim.
    # Use transforms to convert a half-line pixel offset into data units
    fig = ax.figure
    fig.canvas.draw()
    half_line_px = 0.5 * ZOOM_BOX_LW * fig.dpi / 72.0

    # Left inner edge display coord
    disp_left, _ = ax.transData.transform((xlim[0], 0.0))
    disp_right, _ = ax.transData.transform((xlim[1], 0.0))
    # Move the centerline to the right by half the linewidth in display coords
    disp_left_center = disp_left + half_line_px
    disp_right_center = disp_right - half_line_px
    # Convert back to data coords
    x0 = ax.transData.inverted().transform((disp_left_center, 0.0))[0]
    x1 = ax.transData.inverted().transform((disp_right_center, 0.0))[0]

    _, disp_bottom = ax.transData.transform((0.0, ylim[0]))
    _, disp_top = ax.transData.transform((0.0, ylim[1]))
    disp_bottom_center = disp_bottom + half_line_px
    disp_top_center = disp_top - half_line_px
    y0 = ax.transData.inverted().transform((0.0, disp_bottom_center))[1]
    y1 = ax.transData.inverted().transform((0.0, disp_top_center))[1]

    w = max(x1 - x0, 1e-12)
    h = max(y1 - y0, 1e-12)

    rect = Rectangle(
        (x0, y0),
        w,
        h,
        fill=False,
        edgecolor=color,
        linewidth=ZOOM_BOX_LW,
        linestyle="-",
        zorder=4,
    )
    ax.add_patch(rect)


def add_inset(
    ax_main: plt.Axes,
    folder: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    bounds: List[float],
    border_color: str,
) -> bool:
    files = sorted(glob.glob(os.path.join(ZOOM_DATA_ROOT, folder, "*.tsv")))
    if not files:
        return False

    grouped = collect_grouped_files(files)
    ax_inset = ax_main.inset_axes(bounds)
    # Keep inset patch explicitly opaque even when figure background is transparent.
    ax_inset.set_facecolor("white")
    ax_inset.patch.set_facecolor("white")
    ax_inset.patch.set_alpha(1.0)
    ax_inset.patch.set_visible(True)

    plot_grouped_cloud(ax_inset, grouped, add_labels=False)
    ax_inset.set_xlim(*xlim)
    ax_inset.set_ylim(*ylim)

    # Keep grid visible but remove all text elements.
    ax_inset.set_title("")
    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")
    style_axes(ax_inset, show_tick_labels=False, facecolor="white", show_grid=False)

    # Remove the black inset frame; keep only the colored outer border.
    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    # Place a colored border just outside the black frame so they are back-to-back.
    fig = ax_inset.figure
    fig.canvas.draw()
    bbox = ax_inset.get_window_extent()
    axes_w_px = max(bbox.width, 1.0)
    axes_h_px = max(bbox.height, 1.0)

    # Convert linewidth from points to pixels, then to axis-fraction offsets.
    # Use the actual zoom outer linewidth so the inner edge aligns with the inset bounds.
    outer_border_px = ZOOM_OUTER_BORDER_LW * fig.dpi / 72.0
    offset_px = 0.5 * outer_border_px
    dx = offset_px / axes_w_px
    dy = offset_px / axes_h_px

    outer_border = Rectangle(
        (-dx, -dy),
        1.0 + 2.0 * dx,
        1.0 + 2.0 * dy,
        transform=ax_inset.transAxes,
        fill=False,
        edgecolor=border_color,
        linewidth=ZOOM_OUTER_BORDER_LW,
        clip_on=False,
        zorder=7,
    )
    ax_inset.add_patch(outer_border)
    return True


def main() -> None:
    set_plot_style()

    main_files = sorted(glob.glob(os.path.join(MAIN_DATA_DIR, "*.tsv")))
    if not main_files:
        raise RuntimeError(f"No data files found in {MAIN_DATA_DIR}")

    grouped_main = collect_grouped_files(main_files)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    plot_grouped_cloud(ax, grouped_main, add_labels=True)

    hopf = np.pi / 2.0
    ax.axvline(hopf, color=HOPF_COLOR, lw=1.0, ls="--", alpha=0.75, label=r"$k\tau=\pi/2$")

    ax.set_xlabel(r"$k\tau$")
    ax.set_ylabel(r"$\theta_{\max}$")
    ax.set_ylim(0.0, TWO_PI)
    ax.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, TWO_PI])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    style_axes(ax)

    for spec in INSET_SPECS:
        draw_zoom_box(ax, spec["xlim"], spec["ylim"], spec["color"])
        add_inset(ax, spec["folder"], spec["xlim"], spec["ylim"], spec["bounds"], spec["color"])

    os.makedirs(PLOT_DIR, exist_ok=True)
    out_png = os.path.join(PLOT_DIR, "bifurcation_4ic_target_with_insets.png")
    fig.savefig(out_png, format="png", dpi=800, transparent=False, facecolor="none", edgecolor="none")
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
