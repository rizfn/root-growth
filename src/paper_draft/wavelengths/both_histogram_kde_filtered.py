from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from errorbars_filter import analyze_image, filter_rows_by_peak_threshold, set_plot_style, style_axes


COLOR_TILTED = "#901A1E"
COLOR_NON_TILTED = "#547AA5"
COLOR_REFERENCE = "#666666"


def gaussian_kernel_pdf(values, errors, x_grid, sigma_floor=1e-3):
    """Heteroscedastic Gaussian-kernel probability density.

    Each point contributes equal mass 1/N using sigma_i from its uncertainty.
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    if values.size == 0 or x_grid.size == 0:
        return np.zeros_like(x_grid, dtype=float)

    out = np.zeros_like(x_grid, dtype=float)
    floor = float(max(sigma_floor, 1e-6))
    norm = np.sqrt(2.0 * np.pi)

    for mu, sig in zip(values, errors):
        if not np.isfinite(mu):
            continue
        sigma = float(sig) if np.isfinite(sig) and sig > 0 else floor
        sigma = max(floor, sigma)
        z = (x_grid - mu) / sigma
        out += np.exp(-0.5 * z * z) / (sigma * norm)

    return out / float(values.size)


def plot_kernel_distribution(rows, out_svg):
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No valid wavelength estimates to plot.")

    vals = np.array([r["wavelength_firstpeak_mm"] for r in rows if np.isfinite(r["wavelength_firstpeak_mm"])], dtype=float)
    if vals.size == 0:
        print("No valid wavelength estimates to plot.")

    y_tilted = np.array(
        [r["wavelength_firstpeak_mm"] for r in rows if "AgarTilted" in r["image"] and np.isfinite(r["wavelength_firstpeak_mm"])],
        dtype=float,
    )
    e_tilted = np.array(
        [
            float(r["wavelength_firstpeak_err_mm"])
            for r in rows
            if "AgarTilted" in r["image"] and np.isfinite(r["wavelength_firstpeak_mm"])
        ],
        dtype=float,
    )
    y_nontilted = np.array(
        [r["wavelength_firstpeak_mm"] for r in rows if "AgarTilted" not in r["image"] and np.isfinite(r["wavelength_firstpeak_mm"])],
        dtype=float,
    )
    e_nontilted = np.array(
        [
            float(r["wavelength_firstpeak_err_mm"])
            for r in rows
            if "AgarTilted" not in r["image"] and np.isfinite(r["wavelength_firstpeak_mm"])
        ],
        dtype=float,
    )

    p_lo = float(np.min(vals))
    p_hi = float(np.max(vals))
    if p_hi <= p_lo:
        center = float(np.median(vals))
        spread = max(float(np.std(vals)), 1e-3)
        p_lo, p_hi = center - 2.0 * spread, center + 2.0 * spread

    bins = np.linspace(p_lo, p_hi, 22)
    if np.allclose(bins[0], bins[-1]):
        bins = np.linspace(p_lo - 1.0, p_hi + 1.0, 22)

    sigma_floor = 0.5 * float(np.median(np.diff(bins))) if len(bins) > 1 else 1e-3
    sigma_floor = max(sigma_floor, 1e-3)
    x_dense = np.linspace(bins[0], bins[-1], 900)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, constrained_layout=True)
    fig.patch.set_alpha(0.0)
    datasets = [
        ("Tilted", y_tilted, e_tilted, COLOR_TILTED),
        ("Non-tilted", y_nontilted, e_nontilted, COLOR_NON_TILTED),
    ]

    for ax, (title, yy, ee, color) in zip(axes, datasets):
        ax.text(0.5, 0.93, title, transform=ax.transAxes, ha="center", va="top", color=color)
        if yy.size == 0:
            ax.text(0.5, 0.5, "No valid roots", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylabel("KDE density")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
            ax_hist = ax.twinx()
            ax_hist.set_ylabel("Count")
            ax_hist.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=3))
            ax_hist.grid(False)
            style_axes(ax)
            continue

        ax_hist = ax.twinx()
        ax_hist.hist(yy, bins=bins, color=color, alpha=0.28, edgecolor="white", linewidth=0.8, rwidth=1)
        ax_hist.set_ylabel("Count")
        ax_hist.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=3))
        ax_hist.grid(False)

        pdf = gaussian_kernel_pdf(
            yy,
            ee,
            x_grid=x_dense,
            sigma_floor=sigma_floor,
        )
        ax.fill_between(x_dense, 0.0, pdf, color=color, alpha=0.22, linewidth=0)
        ax.plot(x_dense, pdf, color=color, linewidth=2.4, alpha=0.95)

        finite_err = np.isfinite(ee) & (ee > 0)
        if np.any(finite_err):
            yy_w = yy[finite_err]
            ee_w = ee[finite_err]
            w = 1.0 / (ee_w ** 2)
            weighted_mean = float(np.sum(w * yy_w) / np.sum(w))
        else:
            weighted_mean = float(np.mean(yy))
        ax.axvline(weighted_mean, color=COLOR_REFERENCE, linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_ylabel("KDE density")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        style_axes(ax)

    axes[-1].set_xlabel("Estimated wavelength (mm)")

    fig.savefig(out_svg, format="svg", transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def main():
    set_plot_style()
    input_dir = Path("data/col_segmented/edited")
    output_dir = Path(__file__).resolve().parent / "plots" / "histogram"

    dx_pixels = 8
    min_root_length = 200
    max_lag = 200
    peak_threshold = 0.1
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

    filtered_rows = filter_rows_by_peak_threshold(all_rows, peak_threshold=peak_threshold)

    out_svg = output_dir / f"histogram_kde_tilted_vs_nontilted_filtered_ct_{peak_threshold:.2f}.svg"
    plot_kernel_distribution(filtered_rows, out_svg)

    print(f"\nTotal root estimates after peak filter (> {peak_threshold}): {len(filtered_rows)}")
    print(f"Histogram-kernel distribution saved to {out_svg}")


if __name__ == "__main__":
    main()
