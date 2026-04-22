from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage import measure
from skimage.morphology import skeletonize
from scipy import signal
import networkx as nx


COLOR_TILTED = "#901A1E"
COLOR_NON_TILTED = "#547AA5"


def set_plot_style():
    """Apply paper_draft-like plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 28,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 17,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def style_axes(ax):
    """Paper-style axes: transparent face, outward ticks, soft grid."""
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.minorticks_on()
    ax.grid(True, axis="y", which="major", alpha=0.3, linewidth=0.7)
    ax.grid(False, axis="x", which="both")
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=1)


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


def first_positive_peak_lag(ac):
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


def estimate_wavelength_first_peak(theta, dx_mm, max_lag=200):
    """Estimate wavelength from first positive autocorrelation peak."""
    theta = np.asarray(theta)
    if len(theta) < 12:
        return np.nan, None, None

    x = np.unwrap(theta)
    ac = compute_autocorr(x, max_lag=max_lag)
    peak_lag = first_positive_peak_lag(ac)
    if peak_lag is None:
        return np.nan, ac, None

    return float(peak_lag * dx_mm), ac, int(peak_lag)


def first_peak_uncertainty_mm(ac, peak_lag, dx_mm):
    """Estimate first-peak wavelength uncertainty from autocorr peak width.

    Uses half-prominence peak width around the selected first positive peak.
    Falls back to one-bin uncertainty when width cannot be computed.
    """
    if ac is None or peak_lag is None or peak_lag <= 0 or peak_lag >= len(ac):
        return dx_mm

    vals = np.asarray(ac, dtype=float)
    if not np.isfinite(vals[peak_lag]):
        return dx_mm

    try:
        widths, _, _, _ = signal.peak_widths(vals, [peak_lag], rel_height=0.5)
        if len(widths) > 0 and np.isfinite(widths[0]) and widths[0] > 0:
            # Half-width in lag bins mapped to mm, with a conservative floor of one bin.
            return float(max(dx_mm, 0.5 * widths[0] * dx_mm))
    except Exception:
        pass

    return float(dx_mm)


def analyze_image(filepath, dx_pixels=8, min_root_length=200, max_lag=200):
    """Return first-peak wavelength estimates for each valid root in one image."""
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

    estimates = []
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

        wl_first_mm, ac, peak_lag = estimate_wavelength_first_peak(thetas, dx_mm=dx_mm, max_lag=max_lag)
        if not np.isfinite(wl_first_mm):
            continue

        # Low-data uncertainty from first-peak width (fallback to one-bin floor).
        wl_first_err_mm = first_peak_uncertainty_mm(ac, peak_lag, dx_mm)

        estimates.append(
            {
                "image": filepath.name,
                "root_label": int(region.label),
                "wavelength_firstpeak_mm": float(wl_first_mm),
                "wavelength_firstpeak_err_mm": float(wl_first_err_mm),
                "peak_lag_index": int(peak_lag),
                "n_theta_samples": int(len(thetas)),
                "main_length_mm": float(main_length * mm_per_pixel),
            }
        )

    print(f"  Estimated wavelengths for {len(estimates)} roots")
    return estimates


def plot_wavelengths(rows, out_png):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid wavelength estimates", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    # Only two groups are shown: tilted vs non-tilted.
    groups = {
        "Tilted": [r for r in rows if "AgarTilted" in r["image"]],
        "Non-tilted": [r for r in rows if "AgarTilted" not in r["image"]],
    }
    group_order = ["Tilted", "Non-tilted"]
    group_to_x = {name: i for i, name in enumerate(group_order)}
    group_colors = {
        "Tilted": COLOR_TILTED,
        "Non-tilted": COLOR_NON_TILTED,
    }

    y_key, err_key = "wavelength_firstpeak_mm", "wavelength_firstpeak_err_mm"
    marker_style = lambda c: {
        "fmt": "o",
        "color": c,
        "ecolor": c,
        "elinewidth": 1.7,
        "capsize": 4,
        "markersize": 6,
        "alpha": 0.9,
        "zorder": 3,
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    group_text = {}

    for grp in group_order:
        pts = groups[grp]
        if not pts:
            continue

        y_vals = []
        y_errs = []
        for row in pts:
            y = row[y_key]
            if not np.isfinite(y):
                continue
            y_vals.append(float(y))
            e = row[err_key]
            y_errs.append(float(e) if np.isfinite(e) and e > 0 else np.nan)

        if len(y_vals) == 0:
            continue

        y_vals = np.asarray(y_vals, dtype=float)
        y_errs = np.asarray(y_errs, dtype=float)
        # Spread points evenly with jitter width that grows with sample size.
        jitter_half_width = min(0.38, 0.015 * len(y_vals))
        if len(y_vals) == 1:
            x = np.array([group_to_x[grp]], dtype=float)
        else:
            x = group_to_x[grp] + np.linspace(-jitter_half_width, jitter_half_width, len(y_vals))
        ax.errorbar(
            x,
            y_vals,
            yerr=np.where(np.isfinite(y_errs), y_errs, np.nan),
            **marker_style(group_colors[grp]),
        )

        n_grp = int(len(y_vals))
        finite_err = np.isfinite(y_errs) & (y_errs > 0)
        if np.any(finite_err):
            yw = y_vals[finite_err]
            ew = y_errs[finite_err]
            w = 1.0 / (ew ** 2)
            mean_grp = float(np.sum(w * yw) / np.sum(w))
            sem_grp = float(np.sqrt(1.0 / np.sum(w)))
        else:
            # Fallback when no finite positive uncertainties are available.
            mean_grp = float(np.mean(y_vals))
            sem_grp = np.nan
        group_text[grp] = (n_grp, mean_grp, sem_grp)

    ax.set_xticks([group_to_x[g] for g in group_order])
    ax.set_xticklabels(group_order)
    ax.tick_params(axis="x", labelsize=26, which="both", length=0)
    ax.set_ylabel("Estimated wavelength (mm)")
    style_axes(ax)

    y0, y1 = ax.get_ylim()
    y_text = y1 - 0.06 * (y1 - y0)
    for grp in group_order:
        n_grp, mean_grp, sem_grp = group_text.get(grp, (0, np.nan, np.nan))
        if np.isfinite(mean_grp):
            if np.isfinite(sem_grp):
                txt = f"n={n_grp}\n$\\mu_{{}}$={mean_grp:.2f} ± {sem_grp:.2f} mm"
            else:
                txt = f"n={n_grp}\n$\\mu_{{}}$={mean_grp:.2f} mm"
            ax.text(
                group_to_x[grp],
                y_text,
                txt,
                color=group_colors[grp],
                ha="center",
                va="top",
                fontsize=17,
            )

    fig.savefig(out_png, format="png", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def main():
    set_plot_style()
    input_dir = Path("data/col_segmented/edited")
    output_dir = Path(__file__).resolve().parent / "plots" / "errorbars"

    dx_pixels = 8
    min_root_length = 200
    max_lag = 200

    files = sorted(input_dir.glob("*.png"))
    if not files:
        print(f"No PNG files found in {input_dir}")
        return

    all_rows = []
    print(f"Found {len(files)} images to process\n")

    for fp in files:
        rows = analyze_image(
            fp,
            dx_pixels=dx_pixels,
            min_root_length=min_root_length,
            max_lag=max_lag,
        )
        all_rows.extend(rows)

    out_png = output_dir / "wavelength_errorbars_tilted_vs_nontilted.png"
    plot_wavelengths(all_rows, out_png)

    print(f"\nTotal root estimates: {len(all_rows)}")
    print(f"Errorbar plot saved to {out_png}")


if __name__ == "__main__":
    main()
