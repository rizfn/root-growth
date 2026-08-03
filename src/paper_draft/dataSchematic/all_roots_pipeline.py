from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from imageio.v2 import imread
from matplotlib.gridspec import GridSpec
from scipy import signal
from skimage import color
from skimage import measure
from skimage.morphology import skeletonize


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
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
    "#D88C2D",
    "#8C4C8A",
    "#4A8A5B",
]


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=6)
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.25)


def build_skeleton_graph(mask: np.ndarray) -> tuple[nx.Graph | None, list[tuple[int, int]]]:
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
    if graph is None or len(coords) == 0:
        return []

    start = min(coords, key=lambda p: p[0])
    lengths = nx.single_source_shortest_path_length(graph, start)
    if not lengths:
        return []
    farthest = max(lengths, key=lengths.get)
    return nx.shortest_path(graph, source=start, target=farthest)


def compute_theta_timeseries(
    path: list[tuple[int, int]],
    dx_pixels: int,
    start_offset: int = 200,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    pts = np.array(path, dtype=float)
    if pts.shape[0] < 2:
        return None, None

    diffs = np.diff(pts, axis=0)
    seg_d = np.sqrt((diffs**2).sum(axis=1))
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


def load_segmented_image(image_path: Path) -> np.ndarray:
    rgba = imread(image_path)
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB or RGBA image: {image_path}")
    return rgba


def load_grayscale_uint8(image_path: Path) -> np.ndarray:
    image = imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        return (image * 255).astype(np.uint8)

    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def resolve_raw_image_path(segmented_path: Path) -> Path:
    raw_dir = Path("data/col_raw")
    candidates = sorted(raw_dir.glob(f"{segmented_path.stem}.*"))
    if candidates:
        return candidates[0]
    return segmented_path


def mask_from_rgba(rgba: np.ndarray) -> np.ndarray:
    if rgba.shape[2] == 4:
        return rgba[:, :, 3] > 0
    return np.any(rgba[:, :, :3] > 0, axis=2)


def build_colored_root_rgba(labeled: np.ndarray, root_entries: list[dict[str, object]]) -> np.ndarray:
    rgba = np.zeros(labeled.shape + (4,), dtype=float)
    for entry in root_entries:
        label_id = int(entry["label_id"])
        color_rgb = entry["color"]
        mask = labeled == label_id
        rgba[mask, :3] = color_rgb
        rgba[mask, 3] = 1.0
    return rgba


def collect_roots(
    rgba: np.ndarray,
    dx_pixels: int,
    min_root_length: int,
    max_lag: int,
) -> list[dict[str, object]]:
    mask = mask_from_rgba(rgba)
    labeled = measure.label(mask, connectivity=2)
    props = sorted(measure.regionprops(labeled), key=lambda r: r.centroid[1])

    palette_rgb = [tuple(mcolors.to_rgb(hex_color)) for hex_color in ROOT_PALETTE_HEX]
    root_entries: list[dict[str, object]] = []

    for display_index, region in enumerate(props, start=1):
        label_id = int(region.label)
        comp_mask = labeled == label_id
        graph, coords = build_skeleton_graph(comp_mask)
        if graph is None or len(coords) < 2:
            continue

        main_path = find_main_trunk(graph, coords)
        if len(main_path) < 2:
            continue

        pts_main = np.array(main_path, dtype=float)
        seg_d_main = np.sqrt(np.sum(np.diff(pts_main, axis=0) ** 2, axis=1))
        main_length = float(seg_d_main.sum()) if seg_d_main.size > 0 else 0.0
        if main_length < min_root_length:
            continue

        xs_pixels, thetas = compute_theta_timeseries(main_path, dx_pixels, start_offset=min_root_length)
        if xs_pixels is None or thetas is None or len(thetas) < 12:
            root_entries.append(
                {
                    "display_index": display_index,
                    "label_id": label_id,
                    "path": main_path,
                    "color": palette_rgb[(display_index - 1) % len(palette_rgb)],
                    "xs_pixels": None,
                    "thetas": None,
                    "ac": None,
                    "peak_lag": None,
                    "peak_width_mm": None,
                }
            )
            continue

        ac = compute_autocorr(np.unwrap(thetas), max_lag=max_lag)
        peak_lag = first_positive_peak_lag(ac)
        dx_mm = dx_pixels * (100.0 / rgba.shape[0])
        peak_width_mm = first_peak_uncertainty_mm(ac, peak_lag, dx_mm) if peak_lag is not None else float(dx_mm)

        root_entries.append(
            {
                "display_index": display_index,
                "label_id": label_id,
                "path": main_path,
                "color": palette_rgb[(display_index - 1) % len(palette_rgb)],
                "xs_pixels": xs_pixels,
                "thetas": thetas,
                "ac": ac,
                "peak_lag": peak_lag,
                "peak_width_mm": peak_width_mm,
            }
        )

    return root_entries


def plot_top_row(
    fig: plt.Figure,
    gs: GridSpec,
    grayscale: np.ndarray,
    root_rgba: np.ndarray,
    labeled: np.ndarray,
    root_entries: list[dict[str, object]],
) -> None:
    ax_input = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])

    ax_input.imshow(grayscale, cmap="gray", interpolation="nearest")
    ax_input.set_axis_off()
    ax_input.set_aspect("auto")

    ax_overlay.imshow(root_rgba, interpolation="nearest")
    ax_overlay.set_aspect("auto")
    for entry in root_entries:
        label_id = int(entry["label_id"])
        color = entry["color"]
        component_mask = labeled == label_id
        ax_overlay.contour(component_mask.astype(float), levels=[0.5], colors=[color], linewidths=2.0)

    ax_overlay.set_axis_off()


def plot_root_row(
    fig: plt.Figure,
    gs: GridSpec,
    row_index: int,
    entry: dict[str, object],
    image_height_px: int,
) -> None:
    ax_theta = fig.add_subplot(gs[row_index, 0])
    ax_ac = fig.add_subplot(gs[row_index, 1])

    root_color = entry["color"]
    display_index = int(entry["display_index"])
    xs_pixels = entry["xs_pixels"]
    thetas = entry["thetas"]
    ac = entry["ac"]
    peak_lag = entry["peak_lag"]
    peak_width_mm = entry["peak_width_mm"]

    if xs_pixels is None or thetas is None or ac is None or peak_lag is None:
        ax_theta.text(0.5, 0.5, "No valid trunk signal", transform=ax_theta.transAxes, ha="center", va="center")
        ax_ac.text(0.5, 0.5, "No valid trunk signal", transform=ax_ac.transAxes, ha="center", va="center")
        ax_theta.set_axis_off()
        ax_ac.set_axis_off()
        return

    mm_per_pixel = 100.0 / image_height_px
    x_mm = np.asarray(xs_pixels, dtype=float) * mm_per_pixel
    x_mm_display = x_mm - float(x_mm.min())
    theta_vals = np.unwrap(np.asarray(thetas, dtype=float))
    lag_mm = np.arange(len(ac)) * (x_mm_display[1] - x_mm_display[0] if len(x_mm_display) > 1 else 8.0 * mm_per_pixel)

    ax_theta.plot(x_mm_display, theta_vals, color=root_color, linewidth=2.0)
    ax_theta.set_xlabel(r"Arc length $r$ [mm]")
    ax_theta.set_ylabel(r"$\theta(r)$ [rad]")
    style_axis(ax_theta)
    ax_theta.set_xlim(float(x_mm_display.min()), float(x_mm_display.max()))
    ax_theta.set_ylim(float(theta_vals.min()), float(theta_vals.max()))

    ax_ac.plot(lag_mm, ac, color=root_color, linewidth=2.0)
    ax_ac.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    if 0 <= peak_lag < len(ac):
        peak_x = float(lag_mm[peak_lag])
        peak_y = float(ac[peak_lag])
        ax_ac.errorbar(
            peak_x,
            peak_y,
            xerr=float(peak_width_mm) if peak_width_mm is not None else None,
            fmt="o",
            color=root_color,
            ecolor="#666666",
            elinewidth=1.2,
            capsize=3,
            markersize=6,
            zorder=5,
        )
        ax_ac.axvline(peak_x, color="#666666", linestyle=":", linewidth=1.0, alpha=0.9)
    ax_ac.set_xlabel(r"Lag $\Delta r$ [mm]")
    ax_ac.set_ylabel(r"$C(\Delta r)$")
    style_axis(ax_ac)
    ax_ac.set_xlim(float(lag_mm.min()), float(lag_mm.max()))
    ax_ac.set_ylim(-1.0, 1.0)


def render_all_roots_pdf(
    image_path: Path,
    output_dir: Path,
    dx_pixels: int,
    min_root_length: int,
    max_lag: int,
) -> None:
    rgba = load_segmented_image(image_path)
    root_entries = collect_roots(rgba, dx_pixels=dx_pixels, min_root_length=min_root_length, max_lag=max_lag)
    grayscale = load_grayscale_uint8(resolve_raw_image_path(image_path))
    mask = mask_from_rgba(rgba)
    labeled = measure.label(mask, connectivity=2)
    root_rgba = build_colored_root_rgba(labeled, root_entries)

    n_root_rows = max(1, len(root_entries))
    fig_height = 5.2 + 2.65 * n_root_rows
    fig = plt.figure(figsize=(14.6, fig_height), constrained_layout=False)
    fig.patch.set_alpha(0.0)

    gs = GridSpec(
        nrows=n_root_rows + 1,
        ncols=2,
        figure=fig,
        height_ratios=[1.9] + [1.0] * n_root_rows,
        hspace=0.48,
        wspace=0.20,
    )

    plot_top_row(fig, gs, grayscale, root_rgba, labeled, root_entries)

    for row_offset, entry in enumerate(root_entries, start=1):
        plot_root_row(
            fig=fig,
            gs=gs,
            row_index=row_offset,
            entry=entry,
            image_height_px=rgba.shape[0],
        )

    if len(root_entries) == 0:
        ax = fig.add_subplot(gs[1, :])
        ax.text(0.5, 0.5, "No valid roots found", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.04, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def main() -> None:
    input_dir = Path("data/col_segmented/edited")
    output_dir = Path("src/paper_draft/dataSchematic/allRoots")

    files = sorted(input_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    for image_path in files:
        render_all_roots_pdf(
            image_path=image_path,
            output_dir=output_dir,
            dx_pixels=8,
            min_root_length=200,
            max_lag=200,
        )


if __name__ == "__main__":
    main()