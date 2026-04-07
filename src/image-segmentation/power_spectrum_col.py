from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import imageio.v2 as imageio
from skimage import measure
from skimage.morphology import skeletonize
from scipy import signal
import networkx as nx


def build_skeleton_graph(mask):
    """Build a pixel-level graph from a binary mask's skeleton."""
    skel = skeletonize(mask)
    coords = [tuple(p) for p in np.transpose(np.where(skel))]
    if len(coords) == 0:
        return None, skel, []

    G = nx.Graph()
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    skel_set = set(coords)

    for p in coords:
        G.add_node(p)
    for p in coords:
        y, x = p
        for dy, dx in dirs:
            q = (y + dy, x + dx)
            if q in skel_set:
                G.add_edge(p, q, weight=1)

    return G, skel, coords


def find_main_trunk(G, coords):
    """Find main trunk path: topmost pixel to farthest pixel by graph distance."""
    if G is None or len(coords) == 0:
        return []
    start = min(coords, key=lambda p: p[0])
    lengths = nx.single_source_shortest_path_length(G, start)
    far = max(lengths, key=lengths.get)
    return nx.shortest_path(G, source=start, target=far)


def compute_theta_timeseries(path, dx_pixels, start_offset=200):
    """Sample trunk path and compute angle timeseries.

    Parameters
    - path: ordered list of (y,x) pixels along trunk
    - dx_pixels: sampling spacing along arc-length
    - start_offset: ignore the first `start_offset` pixels of arc-length (head)
    """
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
        theta = np.arctan2(dxv, dy)
        thetas.append(theta)

    xs = sample_s[:len(thetas)]
    return xs, np.array(thetas)


def compute_autocorr(x, max_lag=None, min_pairs=4):
    """Compute normalized autocorrelation with unbiased normalization.

    To avoid unreliable large-lag values when there are very few
    sample pairs, lags where (n - lag) < `min_pairs` are returned as NaN
    and trailing NaNs are trimmed from the returned array.
    """
    x = np.asarray(x)
    n = len(x)
    mean, var = x.mean(), x.var()
    if var == 0:
        return np.array([1.0])

    x0 = x - mean
    full = np.correlate(x0, x0, mode="full")
    ac_full = full[n - 1:]
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
    ac = ac[: last + 1]

    return ac


def analyze_single_image(filepath, dx_pixels=8, min_root_length=200, max_lag=200):
    """Analyze all components in a single image.

    `min_root_length` filters out components whose main trunk arc-length is shorter than this (pixels).
    """
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

    components = []
    for region in measure.regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr

        comp_mask = labeled == region.label

        G, skel, coords = build_skeleton_graph(comp_mask)
        if G is None or len(coords) < 2:
            print(f"  Skipping component {region.label} (empty/trivial skeleton)")
            continue

        main_path = find_main_trunk(G, coords)
        if len(main_path) < 2:
            print(f"  Skipping component {region.label} (trunk too short)")
            continue

        pts_main = np.array(main_path, dtype=float)
        seg_d_main = np.sqrt(np.sum(np.diff(pts_main, axis=0) ** 2, axis=1))
        main_length = seg_d_main.sum() if seg_d_main.size > 0 else 0.0
        if main_length < min_root_length:
            print(
                f"  Skipping component {region.label} "
                f"(main trunk length {main_length:.1f} < {min_root_length})"
            )
            continue

        xs, thetas = compute_theta_timeseries(main_path, dx_pixels, start_offset=min_root_length)
        if thetas is None or len(thetas) < 4:
            print(f"  Skipping component {region.label} (insufficient theta samples)")
            continue

        comp_data = {
            "label": region.label,
            "mask": comp_mask,
            "skeleton": skel,
            "graph": G,
            "trunk_path": main_path,
            "coords": coords,
            "xs": xs,
            "xs_mm": xs * mm_per_pixel,
            "thetas": thetas,
            "height": height,
            "height_mm": height * mm_per_pixel,
            "main_length": main_length,
            "main_length_mm": main_length * mm_per_pixel,
            "bbox": region.bbox,
        }
        components.append(comp_data)
        print(
            f"  Component {region.label}: height={height}, "
            f"main_length={main_length:.1f}, theta_samples={len(thetas)}"
        )

    if not components:
        print("  No components met the analysis criteria")
        return None

    autocorrs = []
    for comp in components:
        ac = compute_autocorr(comp["thetas"], max_lag=max_lag)
        comp["autocorr"] = ac
        autocorrs.append(ac)

    max_len = max(len(ac) for ac in autocorrs)
    padded = np.full((len(autocorrs), max_len), np.nan)
    for i, ac in enumerate(autocorrs):
        padded[i, : len(ac)] = ac
    ensemble_ac = np.nanmean(padded, axis=0)

    fs = 1.0 / dx_mm
    for comp in components:
        thetas = comp["thetas"]
        n = len(thetas)
        nperseg = min(max(8, n // 4), 1024)
        freqs, psd = signal.welch(thetas, fs=fs, nperseg=nperseg)
        comp["psd_freqs"] = freqs
        comp["psd"] = psd

    return {
        "filename": filepath.name,
        "components": components,
        "ensemble_ac": ensemble_ac,
        "dx_pixels": dx_pixels,
        "dx_mm": dx_mm,
        "mm_per_pixel": mm_per_pixel,
        "image_height_px": image_height_px,
        "max_lag": max_lag,
    }


def plot_single_image_analysis(results, output_path):
    """Create 4-subplot figure for a single image's analysis."""
    components = results["components"]
    dx_pixels = results["dx_pixels"]
    dx_mm = results["dx_mm"]
    mm_per_pixel = results["mm_per_pixel"]
    image_height_px = results["image_height_px"]
    max_lag = results["max_lag"]
    filename = results["filename"]

    n_comp = len(components)
    if n_comp <= 10:
        colors_list = plt.cm.tab10(np.linspace(0, 1, n_comp))
    elif n_comp <= 20:
        colors_list = plt.cm.tab20(np.linspace(0, 1, n_comp))
    else:
        colors_list = plt.cm.gist_rainbow(np.linspace(0, 1, n_comp))

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(
        f"{filename} - {n_comp} roots analyzed (100 mm / {image_height_px} px = {mm_per_pixel:.4f} mm/px)",
        fontsize=12,
        y=0.995,
    )

    ax1 = plt.subplot(2, 2, 1)
    for i, comp in enumerate(components):
        mask = comp["mask"]
        trunk_path = comp["trunk_path"]
        G = comp["graph"]

        color = colors_list[i]

        trunk_edge_set = set()
        for a, b in zip(trunk_path, trunk_path[1:]):
            trunk_edge_set.add(frozenset((tuple(a), tuple(b))))

        ax1.imshow(mask, cmap=mcolors.ListedColormap(["none", color]), alpha=0.2, interpolation="nearest")

        for u, v in G.edges():
            edge_key = frozenset((u, v))
            if edge_key in trunk_edge_set:
                continue
            uy, ux = u
            vy, vx = v
            ax1.plot([ux, vx], [uy, vy], color=color, linewidth=0.5, alpha=0.5)

        if len(trunk_path) > 0:
            arr = np.array(trunk_path)
            ax1.plot(arr[:, 1], arr[:, 0], color=color, linewidth=2, alpha=1.0, label=f"Root {i + 1}")

    ax1.set_title("All roots: trunk (thick) vs branches (thin)")
    ax1.axis("off")
    if n_comp <= 15:
        ax1.legend(fontsize=8, loc="lower left", framealpha=0.7)

    ax2 = plt.subplot(2, 2, 2)
    for i, comp in enumerate(components):
        xs = comp["xs_mm"]
        thetas_deg = np.degrees(comp["thetas"])
        color = colors_list[i]
        ax2.plot(xs, thetas_deg, color=color, linewidth=1.5, alpha=0.75, label=f"Root {i + 1}")

    ax2.set_xlabel("Distance along trunk (mm)")
    ax2.set_ylabel("theta (deg); 0 = downwards")
    ax2.set_title("Theta vs distance (color = root ID)")
    ax2.grid(True, alpha=0.3)
    if n_comp <= 15:
        ax2.legend(fontsize=8, loc="best", ncol=2, framealpha=0.7)

    ax3 = plt.subplot(2, 2, 3)
    lags = np.arange(max_lag + 1) * dx_mm
    for i, comp in enumerate(components):
        ac = comp["autocorr"]
        color = colors_list[i]
        ax3.plot(lags[: len(ac)], ac, color=color, linewidth=1.5, alpha=0.75, label=f"Root {i + 1}")

    ax3.set_xlabel("Lag (mm)")
    ax3.set_ylabel("Autocorrelation")
    ax3.set_title(f"Individual autocorrelations (n={n_comp})")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color="k", linestyle="--", linewidth=0.7)
    if n_comp <= 15:
        ax3.legend(fontsize=8, loc="best", ncol=2, framealpha=0.7)

    # Plot per-root PSDs directly (no ensemble aggregation).
    ax4 = plt.subplot(2, 2, 4)
    plotted = 0
    for i, comp in enumerate(components):
        freqs = comp["psd_freqs"]
        psd = comp["psd"]
        if len(freqs) > 1:
            ax4.loglog(freqs[1:], psd[1:], color=colors_list[i], linewidth=1.5, alpha=0.8, label=f"Root {i + 1}")
            plotted += 1

    if plotted > 0:
        ax4.set_xlabel("Frequency (1/mm)")
        ax4.set_ylabel("PSD")
        ax4.set_title(f"Individual power spectra (n={plotted})")
        ax4.grid(True, which="both", alpha=0.3)
        if n_comp <= 15:
            ax4.legend(fontsize=8, loc="best", ncol=2, framealpha=0.7)
    else:
        ax4.text(0.5, 0.5, "Insufficient data for PSD", ha="center", va="center")
        ax4.set_title("Individual power spectra")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {output_path}")
    plt.close(fig)


def main():
    input_dir = Path("data/col_segmented/edited")
    output_dir = Path("src/image-segmentation/plots/col_edited_per_image")
    output_dir.mkdir(parents=True, exist_ok=True)

    dx_pixels = 8
    min_root_length = 200
    max_lag = 200

    files = sorted(input_dir.glob("*.png"))

    if not files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(files)} images to process\n")

    processed = 0
    for fp in files:
        results = analyze_single_image(
            fp,
            dx_pixels=dx_pixels,
            min_root_length=min_root_length,
            max_lag=max_lag,
        )
        if results is None:
            continue

        out_filename = fp.stem + "_analysis.png"
        out_path = output_dir / out_filename

        plot_single_image_analysis(results, out_path)
        processed += 1
        print()

    print(f"\nProcessed {processed}/{len(files)} images successfully")
    print(f"Plots saved in {output_dir}")


if __name__ == "__main__":
    main()
