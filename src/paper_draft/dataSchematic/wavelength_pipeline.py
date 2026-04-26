from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from imageio.v2 import imread
from scipy import signal
from skimage import measure
from skimage.morphology import skeletonize


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 26,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


ROOT_PALETTE_HEX = [
    "#547AA5",
    "#CBA810",
    "#901A1E",
    "#B8ABC6",
    "#7DE2D1",
]


def build_skeleton_graph(mask: np.ndarray) -> tuple[nx.Graph | None, list[tuple[int, int]]]:
    """Build a pixel-level graph from a binary mask skeleton."""
    skel = skeletonize(mask)
    coords = [tuple(p) for p in np.transpose(np.where(skel))]
    if len(coords) == 0:
        return None, []

    graph = nx.Graph()
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    skel_set = set(coords)

    for coord in coords:
        graph.add_node(coord)

    for y, x in coords:
        for dy, dx in dirs:
            neighbor = (y + dy, x + dx)
            if neighbor in skel_set:
                graph.add_edge((y, x), neighbor, weight=1)

    return graph, coords


def find_main_trunk(graph: nx.Graph | None, coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find trunk path from topmost pixel to the farthest reachable node."""
    if graph is None or len(coords) == 0:
        return []

    start = min(coords, key=lambda p: p[0])
    lengths = nx.single_source_shortest_path_length(graph, start)
    farthest = max(lengths, key=lengths.get)
    return nx.shortest_path(graph, source=start, target=farthest)


def compute_theta_timeseries(
    path: list[tuple[int, int]],
    dx_pixels: int,
    start_offset: int = 200,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample trunk path and compute orientation series along arc-length."""
    pts = np.array(path, dtype=float)
    if pts.shape[0] < 2:
        return None, None

    diffs = np.diff(pts, axis=0)
    seg_d = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate(([0.0], np.cumsum(seg_d)))
    total_len = cum[-1]

    if total_len <= start_offset + dx_pixels:
        return None, None

    sample_s = np.arange(start_offset, total_len, dx_pixels)
    sample_pts: list[np.ndarray] = []
    for s in sample_s:
        i = np.searchsorted(cum, s) - 1
        i = max(0, min(i, len(seg_d) - 1))
        seg_len = seg_d[i]
        t = (s - cum[i]) / seg_len if seg_len > 0 else 0.0
        sample_pts.append(pts[i] + t * (pts[i + 1] - pts[i]))

    thetas = []
    for i in range(len(sample_pts) - 1):
        p0, p1 = sample_pts[i], sample_pts[i + 1]
        dy, dxv = p1[0] - p0[0], p1[1] - p0[1]
        thetas.append(np.arctan2(dxv, dy))

    xs = sample_s[: len(thetas)]
    return xs, np.array(thetas)


def compute_autocorr(x: np.ndarray, max_lag: int | None = None, min_pairs: int = 4) -> np.ndarray:
    """Compute normalized autocorrelation with unbiased normalization."""
    x = np.asarray(x)
    n = len(x)
    mean, var = x.mean(), x.var()
    if var == 0:
        return np.array([1.0])

    x0 = x - mean
    full = np.correlate(x0, x0, mode="full")
    ac_full = full[n - 1 :]
    max_lag = min(max_lag or n - 1, n - 1)

    ac = np.full((max_lag + 1,), np.nan, dtype=float)
    for lag in range(max_lag + 1):
        pairs = n - lag
        if pairs < min_pairs:
            continue
        ac[lag] = ac_full[lag] / (var * pairs)

    finite = np.isfinite(ac)
    if not finite.any():
        return np.array([np.nan])
    last = np.where(finite)[0].max()
    return ac[: last + 1]


def first_positive_peak_lag(ac: np.ndarray) -> int | None:
    """Return the first positive local autocorrelation peak at lag > 0."""
    if len(ac) < 4:
        return None

    vals = np.asarray(ac, dtype=float)
    peaks, _ = signal.find_peaks(vals[1:])
    if len(peaks) == 0:
        return None

    peaks = peaks + 1
    peaks = peaks[vals[peaks] > 0]
    if len(peaks) == 0:
        return None

    return int(peaks[0])


def first_peak_uncertainty_mm(ac: np.ndarray, peak_lag: int | None, dx_mm: float) -> float:
    """Estimate wavelength uncertainty from first-peak width."""
    if peak_lag is None or peak_lag <= 0 or peak_lag >= len(ac):
        return float(dx_mm)

    vals = np.asarray(ac, dtype=float)
    if not np.isfinite(vals[peak_lag]):
        return float(dx_mm)

    try:
        widths, _, _, _ = signal.peak_widths(vals, [peak_lag], rel_height=0.5)
        if len(widths) > 0 and np.isfinite(widths[0]) and widths[0] > 0:
            return float(max(dx_mm, 0.5 * widths[0] * dx_mm))
    except Exception:
        pass

    return float(dx_mm)


def extract_component_color(rgba: np.ndarray, component_mask: np.ndarray) -> tuple[float, float, float]:
    """Estimate the display color of one segmented component from the source image."""
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else np.ones(rgba.shape[:2], dtype=float)
    valid = component_mask & (alpha > 0)
    if not np.any(valid):
        valid = component_mask

    if not np.any(valid):
        return (0.8, 0.1, 0.1)

    vals = rgb[valid]
    if vals.dtype.kind in ("u", "i"):
        vals = vals.astype(float) / 255.0

    mean_color = np.mean(vals, axis=0)
    return tuple(np.clip(mean_color[:3], 0.0, 1.0))


def style_paper_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.grid(True, which="major", alpha=0.25)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=10)


def set_min_max_ticks(ax: plt.Axes, axis: str, lo: float, hi: float) -> None:
    if np.isclose(lo, hi):
        hi = lo + 1.0
    ticks = [lo, hi]
    labels = [f"{lo:.2f}", f"{hi:.2f}"]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def save_svg(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_colored_root_rgba(labeled: np.ndarray, root_colors: dict[int, tuple[float, float, float]]) -> np.ndarray:
    rgba = np.zeros(labeled.shape + (4,), dtype=float)
    for label_id, color in root_colors.items():
        mask = labeled == label_id
        rgba[mask, :3] = color
        rgba[mask, 3] = 1.0
    return rgba


def plot_input_image(rgba: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    fig.patch.set_alpha(0.0)
    ax.imshow(rgba)
    ax.set_axis_off()
    ax.set_facecolor("none")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    save_svg(fig, output_path)


def plot_all_roots_with_trunks(
    rgba: np.ndarray,
    root_paths: dict[int, list[tuple[int, int]]],
    root_colors: dict[int, tuple[float, float, float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    fig.patch.set_alpha(0.0)
    ax.imshow(rgba)

    for root_label, path in root_paths.items():
        if len(path) < 2:
            continue
        pts = np.array(path)
        color = root_colors[root_label]
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2.2)
        ax.scatter(pts[0, 1], pts[0, 0], s=18, color=color, edgecolors="black", linewidths=0.4, zorder=5)

    ax.set_axis_off()
    ax.set_facecolor("none")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    save_svg(fig, output_path)


def plot_root_timeseries(
    root_label: int,
    xs_pixels: np.ndarray,
    thetas: np.ndarray,
    ac: np.ndarray,
    peak_lag: int,
    peak_width_mm: float,
    mm_per_pixel: float,
    dx_mm: float,
    root_color: tuple[float, float, float],
    output_path: Path,
) -> None:
    fig, (ax_theta, ax_ac) = plt.subplots(1, 2, figsize=(13.0, 4.0), dpi=180, sharex=False)
    fig.patch.set_alpha(0.0)

    x_mm = xs_pixels * mm_per_pixel
    x_mm_display = x_mm - float(x_mm.min())
    theta_vals = np.unwrap(thetas)
    lag_mm = np.arange(len(ac)) * dx_mm

    ax_theta.plot(x_mm_display, theta_vals, color=root_color, linewidth=2.0)
    ax_theta.set_xlabel(r"Arc length $r$ [mm]", labelpad=-20)
    ax_theta.set_ylabel(r"$\theta(r)$ [rad]", labelpad=-40)
    style_paper_axis(ax_theta)
    ax_theta.set_xlim(float(x_mm_display.min()), float(x_mm_display.max()))
    ax_theta.set_ylim(float(theta_vals.min()), float(theta_vals.max()))
    set_min_max_ticks(ax_theta, "x", float(x_mm_display.min()), float(x_mm_display.max()))
    set_min_max_ticks(ax_theta, "y", float(theta_vals.min()), float(theta_vals.max()))

    ax_ac.plot(lag_mm, ac, color=root_color, linewidth=2.0)
    ax_ac.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    if 0 <= peak_lag < len(ac):
        peak_x = float(lag_mm[peak_lag])
        peak_y = float(ac[peak_lag])
        ax_ac.errorbar(
            peak_x,
            peak_y,
            xerr=peak_width_mm,
            fmt="o",
            color=root_color,
            ecolor="#666666",
            elinewidth=1.4,
            capsize=3,
            markersize=7,
            zorder=5,
        )
        ax_ac.axvline(peak_x, color="#666666", linestyle=":", linewidth=1.0, alpha=0.9)
    ax_ac.set_xlabel(r"Lag $\Delta r$ [mm]", labelpad=-20)
    ax_ac.set_ylabel(r"$C(\Delta r)$", labelpad=-40)
    style_paper_axis(ax_ac)
    ax_ac.set_xlim(float(lag_mm.min()), float(lag_mm.max()))
    ax_ac.set_ylim(-1.0, 1.0)
    set_min_max_ticks(ax_ac, "x", float(lag_mm.min()), float(lag_mm.max()))
    set_min_max_ticks(ax_ac, "y", -1.0, 1.0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=2.2)
    save_svg(fig, output_path)


def analyze_single_image_pipeline(
    image_path: Path,
    output_dir: Path,
    dx_pixels: int,
    min_root_length: int,
    max_lag: int,
) -> None:
    rgba = imread(image_path)
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA segmented image.")

    image_height_px = rgba.shape[0]
    mm_per_pixel = 100.0 / image_height_px
    dx_mm = dx_pixels * mm_per_pixel

    if rgba.shape[2] == 4:
        mask = rgba[:, :, 3] > 0
    else:
        mask = np.any(rgba[:, :, :3] > 0, axis=2)

    labeled = measure.label(mask, connectivity=2)

    root_colors: dict[int, tuple[float, float, float]] = {}
    root_paths: dict[int, list[tuple[int, int]]] = {}

    props = list(measure.regionprops(labeled))
    labels_left_to_right = [int(region.label) for region in sorted(props, key=lambda r: r.centroid[1])]
    palette_rgb = [tuple(mcolors.to_rgb(hex_color)) for hex_color in ROOT_PALETTE_HEX]
    if labels_left_to_right:
        root_colors = {
            label: palette_rgb[i % len(palette_rgb)]
            for i, label in enumerate(labels_left_to_right)
        }

    for region in props:
        root_label = int(region.label)
        comp_mask = labeled == root_label

        graph, coords = build_skeleton_graph(comp_mask)
        if graph is None or len(coords) < 2:
            continue

        main_path = find_main_trunk(graph, coords)
        if len(main_path) < 2:
            continue

        pts_main = np.array(main_path, dtype=float)
        seg_d_main = np.sqrt(np.sum(np.diff(pts_main, axis=0) ** 2, axis=1))
        main_length = seg_d_main.sum() if seg_d_main.size > 0 else 0.0
        if main_length < min_root_length:
            continue

        root_paths[root_label] = main_path

    recolored_rgba = build_colored_root_rgba(labeled, root_colors)

    plot_input_image(recolored_rgba, output_dir / "01_input_image.svg")
    plot_all_roots_with_trunks(recolored_rgba, root_paths, root_colors, output_dir / "02_all_roots_with_trunks.svg")

    for root_label in sorted(root_paths):
        main_path = root_paths[root_label]
        root_color = root_colors[root_label]

        xs_pixels, thetas = compute_theta_timeseries(main_path, dx_pixels, start_offset=min_root_length)
        if xs_pixels is None or thetas is None or len(thetas) < 12:
            continue

        ac = compute_autocorr(np.unwrap(thetas), max_lag=max_lag)
        peak_lag = first_positive_peak_lag(ac)
        if peak_lag is None:
            continue

        _wavelength_mm = float(peak_lag * dx_mm)
        wavelength_err_mm = first_peak_uncertainty_mm(ac, peak_lag, dx_mm)

        plot_root_timeseries(
            root_label=root_label,
            xs_pixels=xs_pixels,
            thetas=thetas,
            ac=ac,
            peak_lag=peak_lag,
            peak_width_mm=wavelength_err_mm,
            mm_per_pixel=mm_per_pixel,
            dx_mm=dx_mm,
            root_color=root_color,
            output_path=output_dir / f"03_root{root_label:02d}_timeseries.svg",
        )


def main() -> None:
    analyze_single_image_pipeline(
        image_path=Path("data/col_segmented/edited/Col-0_20220630112924-0001_1.5AgarVertical.png"),
        output_dir=Path("src/paper_draft/dataSchematic/plots/wavelength_pipeline"),
        dx_pixels=8,
        min_root_length=200,
        max_lag=200,
    )


if __name__ == "__main__":
    main()
