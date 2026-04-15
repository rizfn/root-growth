from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import signal
from skimage import measure
from skimage.morphology import skeletonize


def build_skeleton_graph(mask):
    """Build a pixel-level graph from a binary mask skeleton."""
    skel = skeletonize(mask)
    coords = [tuple(p) for p in np.transpose(np.where(skel))]
    if len(coords) == 0:
        return None, skel, []

    graph = nx.Graph()
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    skel_set = set(coords)

    for p in coords:
        graph.add_node(p)
    for p in coords:
        y, x = p
        for dy, dx in dirs:
            q = (y + dy, x + dx)
            if q in skel_set:
                graph.add_edge(p, q, weight=1)

    return graph, skel, coords


def find_main_trunk(graph, coords):
    """Find trunk path from topmost pixel to farthest graph node."""
    if graph is None or len(coords) == 0:
        return []
    start = min(coords, key=lambda p: p[0])
    lengths = nx.single_source_shortest_path_length(graph, start)
    far = max(lengths, key=lengths.get)
    return nx.shortest_path(graph, source=start, target=far)


def compute_theta_timeseries(path, dx_pixels, start_offset=200):
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
    sample_pts = []
    for s in sample_s:
        i = np.searchsorted(cum, s) - 1
        i = max(0, min(i, len(seg_d) - 1))
        seg_len = seg_d[i]
        t = (s - cum[i]) / seg_len if seg_len > 0 else 0.0
        p = pts[i] + t * (pts[i + 1] - pts[i])
        sample_pts.append(p)

    thetas = []
    for i in range(len(sample_pts) - 1):
        p0, p1 = sample_pts[i], sample_pts[i + 1]
        dy, dxv = p1[0] - p0[0], p1[1] - p0[1]
        thetas.append(np.arctan2(dxv, dy))

    xs = sample_s[: len(thetas)]
    return xs, np.array(thetas)


def compute_autocorr(x, max_lag=None, min_pairs=4):
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


def thresholded_peak_candidates(ac, dx_mm, threshold, min_distance=2):
    """Return all positive-lag autocorrelation peaks above the threshold."""
    if ac is None or len(ac) < 3:
        return []

    vals = np.asarray(ac, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return []

    peaks, props = signal.find_peaks(vals[1:], height=threshold, distance=min_distance)
    if len(peaks) == 0:
        return []

    heights = np.asarray(props["peak_heights"], dtype=float)
    rows = []
    for rank, (peak_idx, peak_height) in enumerate(zip(peaks + 1, heights), start=1):
        rows.append(
            {
                "peak_rank": int(rank),
                "peak_lag_index": int(peak_idx),
                "peak_lag_mm": float(peak_idx * dx_mm),
                "autocorr_peak_height": float(peak_height),
            }
        )
    return rows


def compute_candidate_rows_for_thresholds(ac, dx_mm, thresholds, min_distance=2):
    """Return thresholded peak rows for every threshold in a sweep."""
    all_rows = []
    for threshold in thresholds:
        peak_rows = thresholded_peak_candidates(ac, dx_mm=dx_mm, threshold=threshold, min_distance=min_distance)
        for peak_row in peak_rows:
            peak_row = dict(peak_row)
            peak_row["threshold"] = float(threshold)
            all_rows.append(peak_row)
    return all_rows


def smooth_probability_density(values, grid, bandwidth):
    """Estimate a smooth, normalized probability density from discrete values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(grid, dtype=float)

    bandwidth = float(max(bandwidth, 1e-6))
    diffs = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs ** 2).sum(axis=1)
    area = float(np.trapezoid(density, grid))
    if area > 0:
        density /= area
    return density


def plot_file_threshold_distributions(root_entries, out_png, title, dx_mm):
    """Plot wavelength density curves for all roots in one file."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not root_entries:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.text(0.5, 0.5, "No wavelength candidates above the tested thresholds", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    all_values = []
    for root_entry in root_entries:
        for threshold_values in root_entry["wavelengths_by_threshold"].values():
            all_values.extend(threshold_values)

    all_values = np.asarray(all_values, dtype=float)
    finite_values = all_values[np.isfinite(all_values)]
    x_min = float(np.min(finite_values))
    x_max = float(np.max(finite_values))
    span = max(x_max - x_min, dx_mm)
    pad = max(2.0 * dx_mm, 0.15 * span)
    grid = np.linspace(x_min - pad, x_max + pad, 600)

    n_roots = len(root_entries)
    n_cols = 2 if n_roots > 1 else 1
    n_rows = int(np.ceil(n_roots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.8 * n_rows), squeeze=False)

    thresholds = sorted({float(t) for root_entry in root_entries for t in root_entry["wavelengths_by_threshold"].keys()})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds))) if thresholds else []
    threshold_to_color = {threshold: color for threshold, color in zip(thresholds, colors)}

    for index, root_entry in enumerate(root_entries):
        ax = axes[index // n_cols][index % n_cols]
        wavelengths_by_threshold = root_entry["wavelengths_by_threshold"]

        for threshold in thresholds:
            values = np.asarray(wavelengths_by_threshold.get(threshold, []), dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            bandwidth = max(0.2 * dx_mm, 0.04 * span)
            density = smooth_probability_density(values, grid, bandwidth=bandwidth)
            ax.plot(grid, density, color=threshold_to_color[threshold], linewidth=1.7, label=f"thr={threshold:.2f} (n={values.size})")

        ax.set_title(f"root {root_entry['root_label']}")
        ax.set_xlabel("Wavelength candidate (mm)")
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, framealpha=0.85, loc="best")

    for index in range(n_roots, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_axis_off()

    fig.suptitle(title, y=1.01)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_threshold_distribution(wavelengths_by_threshold, out_png, title):
    """Plot one aggregate wavelength density curve per threshold across all images and roots."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    all_values = []
    for values in wavelengths_by_threshold.values():
        all_values.extend(values)
    all_values = np.asarray(all_values, dtype=float)
    finite_all = all_values[np.isfinite(all_values)]

    if finite_all.size == 0:
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        ax.text(0.5, 0.5, "No aggregate wavelength candidates above the tested thresholds", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    x_min = float(np.min(finite_all))
    x_max = float(np.max(finite_all))
    span = max(x_max - x_min, 1e-6)
    pad = max(0.15 * span, 0.3)
    grid = np.linspace(x_min - pad, x_max + pad, 700)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    thresholds = sorted(float(t) for t in wavelengths_by_threshold.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds))) if thresholds else []

    for threshold, color in zip(thresholds, colors):
        values = np.asarray(wavelengths_by_threshold.get(threshold, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        bandwidth = max(0.03 * span, 0.15)
        density = smooth_probability_density(values, grid, bandwidth=bandwidth)
        ax.plot(grid, density, color=color, linewidth=2.0, label=f"thr={threshold:.2f} (n={values.size})")

    ax.set_title(title)
    ax.set_xlabel("Wavelength candidate (mm)")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8, framealpha=0.88, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_image(filepath, dx_pixels=8, min_root_length=200, max_lag=200, thresholds=(0.25,), output_dir=None):
    """Save one threshold-distribution plot per input image."""
    print(f"Processing {filepath.name}")
    rgba = imageio.imread(filepath)
    image_height_px = rgba.shape[0]
    mm_per_pixel = 100.0 / image_height_px
    dx_mm = dx_pixels * mm_per_pixel

    if rgba.shape[2] == 4:
        mask = rgba[:, :, 3] > 0
    else:
        mask = np.any(rgba[:, :, :3] > 0, axis=2)

    labeled = measure.label(mask, connectivity=2)

    root_entries = []
    for region in measure.regionprops(labeled):
        comp_mask = labeled == region.label

        graph, _, coords = build_skeleton_graph(comp_mask)
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

        _, thetas = compute_theta_timeseries(main_path, dx_pixels, start_offset=min_root_length)
        if thetas is None or len(thetas) < 12:
            continue

        ac = compute_autocorr(thetas, max_lag=max_lag)
        peak_rows = compute_candidate_rows_for_thresholds(ac, dx_mm=dx_mm, thresholds=thresholds)
        if not peak_rows:
            continue
        threshold_to_wavelengths = {}
        for threshold in thresholds:
            threshold_to_wavelengths[float(threshold)] = [
                peak_row["peak_lag_mm"]
                for peak_row in peak_rows
                if np.isclose(peak_row["threshold"], threshold)
            ]

        root_entries.append(
            {
                "root_label": int(region.label),
                "wavelengths_by_threshold": threshold_to_wavelengths,
            }
        )

    if output_dir is not None and root_entries:
        out_png = output_dir / f"{filepath.stem}_threshold_distribution.png"
        plot_file_threshold_distributions(
            root_entries,
            out_png=out_png,
            title=f"{filepath.name} | threshold sweep across {len(root_entries)} roots",
            dx_mm=dx_mm,
        )
        print(f"  Saved 1 file-level threshold plot")
        return [str(out_png)], root_entries

    print(f"  No valid roots found for plotting")
    return [], root_entries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate wavelength candidates from autocorrelation peaks above a threshold."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/col_segmented/edited"),
        help="Directory with segmented PNG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/image-segmentation/plots/wavelength_threshold_distribution_col"),
        help="Directory for plots.",
    )
    parser.add_argument("--dx-pixels", type=int, default=8, help="Arc-length sampling spacing in pixels.")
    parser.add_argument(
        "--min-root-length",
        type=int,
        default=200,
        help="Skip roots with main trunk shorter than this length in pixels.",
    )
    parser.add_argument("--max-lag", type=int, default=200, help="Maximum autocorrelation lag to inspect.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4],
        help="One or more autocorrelation peak-height thresholds to sweep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files = sorted(args.input_dir.glob("*.png"))
    if not files:
        print(f"No PNG files found in {args.input_dir}")
        return

    print(f"Found {len(files)} images to process\n")

    saved_plots = []
    aggregate_wavelengths_by_threshold = {float(t): [] for t in args.thresholds}
    total_roots_used = 0

    for fp in files:
        plots, root_entries = analyze_image(
            fp,
            dx_pixels=args.dx_pixels,
            min_root_length=args.min_root_length,
            max_lag=args.max_lag,
            thresholds=args.thresholds,
            output_dir=args.output_dir,
        )
        saved_plots.extend(plots)

        total_roots_used += len(root_entries)
        for root_entry in root_entries:
            for threshold, values in root_entry["wavelengths_by_threshold"].items():
                aggregate_wavelengths_by_threshold[float(threshold)].extend(values)

    has_aggregate_values = any(len(values) > 0 for values in aggregate_wavelengths_by_threshold.values())
    if has_aggregate_values:
        aggregate_out_png = args.output_dir / "all_images_all_roots_threshold_distribution.png"
        plot_aggregate_threshold_distribution(
            aggregate_wavelengths_by_threshold,
            out_png=aggregate_out_png,
            title=(
                f"All images | all roots ({total_roots_used} roots) | "
                f"threshold sweep"
            ),
        )
        saved_plots.append(str(aggregate_out_png))
        print("Saved aggregate all-images/all-roots threshold plot")
    else:
        print("No aggregate wavelength candidates found across all images")

    print(f"\nTotal root plots saved: {len(saved_plots)}")
    if saved_plots:
        print(f"Plots saved under {args.output_dir}")


if __name__ == "__main__":
    main()