from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage import measure
from skimage.morphology import skeletonize
from scipy import signal
from scipy.optimize import least_squares
import networkx as nx


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


def fit_wavelength_from_autocorr(ac, dx_mm, first_peak_lag=None):
    """Estimate wavelength by fitting a robust (non-damped) cosine to autocorrelation.

    Model: c + A * cos(2*pi*lag/lambda + phi)
    """
    if ac is None or len(ac) < 10:
        return np.nan, np.nan, False, np.nan, None

    lags_mm = np.arange(len(ac)) * dx_mm

    # Skip lag=0 (trivial value near 1) and focus on short/intermediate lags.
    start_idx = 1
    end_idx = max(start_idx + 8, int(0.65 * len(ac)))
    end_idx = min(end_idx, len(ac))

    x = lags_mm[start_idx:end_idx]
    y = np.asarray(ac[start_idx:end_idx], dtype=float)
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(y) < 8:
        return np.nan, np.nan, False, np.nan, None

    max_lag_mm = float(np.max(x))
    if first_peak_lag is not None and first_peak_lag > 0:
        lambda0 = first_peak_lag * dx_mm
    else:
        lambda0 = max(4 * dx_mm, 0.25 * max_lag_mm)

    lambda0 = float(np.clip(lambda0, 2.5 * dx_mm, 0.9 * max_lag_mm))
    A0 = float(0.5 * (np.nanmax(y) - np.nanmin(y)))
    phi0 = 0.0
    c0 = float(np.nanmean(y))

    def model(p, xx):
        A, lam, phi, c = p
        return c + A * np.cos(2.0 * np.pi * xx / lam + phi)

    def residuals(p):
        return model(p, x) - y

    p0 = np.array([A0, lambda0, phi0, c0], dtype=float)
    lb = np.array([0.0, 2.0 * dx_mm, -2.0 * np.pi, -1.5], dtype=float)
    ub = np.array([2.0, 0.98 * max_lag_mm, 2.0 * np.pi, 1.5], dtype=float)

    try:
        res = least_squares(residuals, p0, bounds=(lb, ub), loss="soft_l1", f_scale=0.05, max_nfev=3000)
    except Exception:
        return np.nan, np.nan, False, np.nan, None

    if (not res.success) or (not np.all(np.isfinite(res.x))):
        return np.nan, np.nan, False, np.nan, None

    fit_rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    lambda_fit = float(res.x[1])

    # Approximate 1-sigma uncertainty for lambda from local Jacobian covariance.
    lambda_err = np.nan
    try:
        m = len(y)
        p = len(res.x)
        if m > p:
            rss = float(np.sum(res.fun ** 2))
            sigma2 = rss / (m - p)
            jtj_inv = np.linalg.pinv(res.jac.T @ res.jac)
            cov = sigma2 * jtj_inv
            lambda_err = float(np.sqrt(max(0.0, cov[1, 1])))
    except Exception:
        lambda_err = np.nan

    return lambda_fit, lambda_err, True, fit_rmse, np.array(res.x, dtype=float)


def plot_autocorr_diagnostic(
    ac,
    peak_lag,
    dx_mm,
    output_path,
    title,
    fit_params=None,
    fit_ok=False,
    first_peak_err_mm=np.nan,
):
    """Plot autocorrelation with first-peak marker."""
    if ac is None or len(ac) < 2:
        return False

    lags_mm = np.arange(len(ac)) * dx_mm

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lags_mm, ac, color="0.2", linewidth=1.6, label="Autocorrelation")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

    if peak_lag is not None and peak_lag < len(ac):
        if np.isfinite(first_peak_err_mm) and first_peak_err_mm > 0:
            ax.errorbar(
                lags_mm[peak_lag],
                ac[peak_lag],
                xerr=first_peak_err_mm,
                fmt="o",
                color="tab:red",
                ecolor="tab:red",
                elinewidth=1.0,
                capsize=2,
                markersize=7,
                label="First positive peak",
            )
        else:
            ax.plot(lags_mm[peak_lag], ac[peak_lag], "o", color="tab:red", markersize=7, label="First positive peak")
        ax.axvline(lags_mm[peak_lag], color="tab:red", linestyle=":", linewidth=1.0, alpha=0.8)

    if fit_ok and fit_params is not None and np.all(np.isfinite(fit_params)):
        A, lam, phi, c = fit_params
        if lam > 0:
            y_fit = c + A * np.cos(2.0 * np.pi * lags_mm / lam + phi)
            ax.plot(lags_mm, y_fit, color="tab:blue", linestyle="--", linewidth=1.3, alpha=0.95, label="Sinusoid fit")

    ax.set_xlabel("Lag (mm)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def analyze_image(filepath, dx_pixels=8, min_root_length=200, max_lag=200, ac_diag_dir=None):
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

        wl_fit_mm, wl_fit_err_mm, fit_ok, fit_rmse, fit_params = fit_wavelength_from_autocorr(
            ac,
            dx_mm=dx_mm,
            first_peak_lag=peak_lag,
        )
        # Low-data uncertainty from first-peak width (fallback to one-bin floor).
        wl_first_err_mm = first_peak_uncertainty_mm(ac, peak_lag, dx_mm)

        if ac_diag_dir is not None:
            diag_name = f"{filepath.stem}_root{int(region.label):02d}_autocorr.png"
            diag_path = ac_diag_dir / diag_name
            plot_autocorr_diagnostic(
                ac,
                peak_lag=peak_lag,
                dx_mm=dx_mm,
                output_path=diag_path,
                title=f"{filepath.name} | root {int(region.label)}",
                fit_params=fit_params,
                fit_ok=fit_ok,
                first_peak_err_mm=wl_first_err_mm,
            )

        estimates.append(
            {
                "image": filepath.name,
                "root_label": int(region.label),
                "wavelength_firstpeak_mm": float(wl_first_mm),
                "wavelength_firstpeak_err_mm": float(wl_first_err_mm),
                "wavelength_fit_mm": float(wl_fit_mm) if np.isfinite(wl_fit_mm) else np.nan,
                "wavelength_fit_err_mm": float(wl_fit_err_mm) if np.isfinite(wl_fit_err_mm) else np.nan,
                "fit_success": bool(fit_ok),
                "fit_rmse": float(fit_rmse) if np.isfinite(fit_rmse) else np.nan,
                "peak_lag_index": int(peak_lag),
                "n_theta_samples": int(len(thetas)),
                "main_length_mm": float(main_length * mm_per_pixel),
            }
        )

    print(f"  Estimated wavelengths for {len(estimates)} roots")
    return estimates


def write_csv(rows, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "root_label",
        "wavelength_firstpeak_mm",
        "wavelength_firstpeak_err_mm",
        "wavelength_fit_mm",
        "wavelength_fit_err_mm",
        "fit_success",
        "fit_rmse",
        "peak_lag_index",
        "n_theta_samples",
        "main_length_mm",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_wavelengths(rows, out_png, method="firstpeak"):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid wavelength estimates", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    image_names = sorted({r["image"] for r in rows})
    image_to_x = {name: i for i, name in enumerate(image_names)}

    if len(image_names) <= 10:
        colors_list = plt.cm.tab10(np.linspace(0, 1, len(image_names)))
    elif len(image_names) <= 20:
        colors_list = plt.cm.tab20(np.linspace(0, 1, len(image_names)))
    else:
        colors_list = plt.cm.gist_rainbow(np.linspace(0, 1, len(image_names)))
    image_to_color = {name: colors_list[i] for i, name in enumerate(image_names)}

    grouped = {name: [] for name in image_names}
    for row in rows:
        grouped[row["image"]].append(row)

    if method == "firstpeak":
        y_key, err_key = "wavelength_firstpeak_mm", "wavelength_firstpeak_err_mm"
        marker_style = lambda c, lbl: {
            "fmt": "o",
            "color": c,
            "ecolor": c,
            "elinewidth": 1.7,
            "capsize": 4,
            "markersize": 6,
            "alpha": 0.9,
            "zorder": 3,
            "label": lbl,
        }
    else:
        y_key, err_key = "wavelength_fit_mm", "wavelength_fit_err_mm"
        marker_style = lambda c, lbl: {
            "fmt": "o",
            "markersize": 6,
            "markerfacecolor": "none",
            "markeredgecolor": c,
            "markeredgewidth": 1.3,
            "ecolor": c,
            "elinewidth": 1.7,
            "capsize": 4,
            "alpha": 0.95,
            "zorder": 4,
            "label": lbl,
        }

    def legend_label(dataset_rows):
        y = np.array([r[y_key] for r in dataset_rows if np.isfinite(r[y_key])], dtype=float)
        if y.size == 0:
            return "n/a"
        yw = np.array([r[y_key] for r in dataset_rows if np.isfinite(r[y_key]) and np.isfinite(r[err_key]) and r[err_key] > 0], dtype=float)
        ye = np.array([r[err_key] for r in dataset_rows if np.isfinite(r[y_key]) and np.isfinite(r[err_key]) and r[err_key] > 0], dtype=float)
        if yw.size > 0:
            w = 1.0 / (ye ** 2)
            if np.isfinite(np.sum(w)) and np.sum(w) > 0:
                return f"{float(np.sum(w * yw) / np.sum(w)):.2f} mm"
        return f"{float(np.mean(y)):.2f} mm"

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(image_names)), 6))

    legend_done = set()
    for img in image_names:
        pts = grouped[img]
        n = len(pts)
        jitter = np.array([0.0]) if n == 1 else np.linspace(-0.25, 0.25, n)
        lbl = legend_label(pts)

        for j, row in enumerate(pts):
            x = image_to_x[img] + jitter[j]
            color = image_to_color[img]
            label = lbl if img not in legend_done else None

            y = row[y_key]
            yerr = row[err_key]

            if method == "fit" and not np.isfinite(y):
                legend_done.add(img)
                continue

            ax.errorbar(
                x,
                y,
                yerr=yerr if np.isfinite(yerr) else None,
                **marker_style(color, label),
            )

            legend_done.add(img)

    ax.set_xticks(range(len(image_names)))
    ax.set_xticklabels(image_names, rotation=35, ha="right")
    ax.set_ylabel("Estimated wavelength (mm)")
    ax.set_xlabel("Image")
    if method == "firstpeak":
        ax.set_title("Per-root wavelength estimates from first autocorrelation peak\n(color = image)")
    else:
        ax.set_title("Per-root wavelength estimates from sinusoid fit to autocorrelation\n(color = image)")
        fit_y = np.array([r["wavelength_fit_mm"] for r in rows if np.isfinite(r["wavelength_fit_mm"])], dtype=float)
        if fit_y.size > 0:
            fit_yerr = np.array(
                [r["wavelength_fit_err_mm"] for r in rows if np.isfinite(r["wavelength_fit_mm"]) and np.isfinite(r["wavelength_fit_err_mm"]) and r["wavelength_fit_err_mm"] > 0],
                dtype=float,
            )
            y_lo, y_hi = float(np.percentile(fit_y, 5)), float(np.percentile(fit_y, 95))
            if y_hi <= y_lo:
                center = float(np.median(fit_y))
                spread = max(float(np.std(fit_y)), 1e-3)
                y_lo, y_hi = center - spread, center + spread
            typical_err = float(np.percentile(fit_yerr, 75)) if fit_yerr.size > 0 else 0.0
            pad = max(0.08 * (y_hi - y_lo), 1e-3)
            ax.set_ylim(y_lo - typical_err - pad, y_hi + typical_err + pad)
    ax.grid(True, axis="y", alpha=0.25)

    if len(image_names) <= 20:
        ax.legend(fontsize=8, loc="best", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    input_dir = Path("data/col_segmented/edited")
    output_dir = Path("src/image-segmentation/plots/wavelength_estimates_col")

    dx_pixels = 8
    min_root_length = 200
    max_lag = 200

    out_ac_dir = output_dir / "autocorr_diagnostics"
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
            ac_diag_dir=out_ac_dir,
        )
        all_rows.extend(rows)

    out_csv = output_dir / "wavelength_estimates.csv"
    out_png_first = output_dir / "wavelength_estimates_firstpeak.png"
    out_png_fit = output_dir / "wavelength_estimates_fit.png"

    write_csv(all_rows, out_csv)
    plot_wavelengths(all_rows, out_png_first, method="firstpeak")
    plot_wavelengths(all_rows, out_png_fit, method="fit")

    print(f"\nTotal root estimates: {len(all_rows)}")
    print(f"CSV saved to {out_csv}")
    print(f"First-peak plot saved to {out_png_first}")
    print(f"Sinusoid-fit plot saved to {out_png_fit}")
    print(f"Autocorrelation diagnostics saved in {out_ac_dir}")


if __name__ == "__main__":
    main()
