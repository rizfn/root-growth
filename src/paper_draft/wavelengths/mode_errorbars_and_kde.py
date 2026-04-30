from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from errorbars_filter import analyze_image, filter_rows_by_peak_threshold, set_plot_style, style_axes


COLOR_TILTED = "#901A1E"
COLOR_NON_TILTED = "#547AA5"
COLOR_REFERENCE = "#666666"
COLOR_YELLOW = "#CBA810"


def _hsm_recursive(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        return np.nan
    if n == 1:
        return float(sorted_vals[0])
    if n == 2:
        return float(0.5 * (sorted_vals[0] + sorted_vals[1]))
    if n == 3:
        left = sorted_vals[1] - sorted_vals[0]
        right = sorted_vals[2] - sorted_vals[1]
        if left < right:
            return float(0.5 * (sorted_vals[0] + sorted_vals[1]))
        if right < left:
            return float(0.5 * (sorted_vals[1] + sorted_vals[2]))
        return float(sorted_vals[1])

    k = (n + 1) // 2
    widths = sorted_vals[k - 1 :] - sorted_vals[: n - k + 1]
    i_min = int(np.argmin(widths))
    return _hsm_recursive(sorted_vals[i_min : i_min + k])


def compute_half_sample_mode(data):
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(_hsm_recursive(np.sort(x)))


def bootstrap_mode_ci(data, n_bootstrap=2000, ci=99.0, random_seed=42, smooth_sigma=0.0):
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return {
            "mode": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "bootstrap_modes": np.array([], dtype=float),
            "ci_percent": float(ci),
            "n_bootstrap": int(n_bootstrap),
        }

    rng = np.random.default_rng(random_seed)
    mode = compute_half_sample_mode(x)
    bs = np.empty((n_bootstrap,), dtype=float)
    smooth_sigma = float(max(0.0, smooth_sigma))

    for i in range(n_bootstrap):
        sample = rng.choice(x, size=n, replace=True)
        if smooth_sigma > 0.0:
            sample = sample + rng.normal(0.0, smooth_sigma, size=n)
        bs[i] = compute_half_sample_mode(sample)

    alpha = (100.0 - float(ci)) / 2.0
    q_lo, q_hi = np.percentile(bs, [alpha, 100.0 - alpha])
    return {
        "mode": float(mode),
        "ci_lower": float(q_lo),
        "ci_upper": float(q_hi),
        "bootstrap_modes": bs,
        "ci_percent": float(ci),
        "n_bootstrap": int(n_bootstrap),
    }


def split_groups(rows):
    return {
        "Tilted": [r for r in rows if "AgarTilted" in str(r.get("image", ""))],
        "Non-tilted": [r for r in rows if "AgarTilted" not in str(r.get("image", ""))],
    }


def wavelengths_from_rows(rows, key="wavelength_firstpeak_mm"):
    vals = []
    for row in rows:
        value = float(row.get(key, np.nan))
        if np.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=float)


def errors_from_rows(rows, key="wavelength_firstpeak_err_mm"):
    vals = []
    for row in rows:
        value = float(row.get(key, np.nan))
        if np.isfinite(value) and value > 0:
            vals.append(value)
    return np.asarray(vals, dtype=float)


def plot_errorbars(rows, out_svg, group_stats):
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid wavelength estimates", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_svg, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    groups = {
        "Tilted": [r for r in rows if "AgarTilted" in r["image"]],
        "Non-tilted": [r for r in rows if "AgarTilted" not in r["image"]],
    }
    group_order = ["Tilted", "Non-tilted"]
    group_to_x = {name: i for i, name in enumerate(group_order)}
    group_colors = {"Tilted": COLOR_TILTED, "Non-tilted": COLOR_NON_TILTED}

    fig, ax = plt.subplots(1, 1, figsize=(8, 7.2), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    for grp in group_order:
        pts = groups[grp]
        if not pts:
            continue

        y_vals = []
        y_errs = []
        for row in pts:
            y = float(row.get("wavelength_firstpeak_mm", np.nan))
            if not np.isfinite(y):
                continue
            y_vals.append(y)
            e = float(row.get("wavelength_firstpeak_err_mm", np.nan))
            y_errs.append(e if np.isfinite(e) and e > 0 else np.nan)

        if len(y_vals) == 0:
            continue

        y_vals = np.asarray(y_vals, dtype=float)
        y_errs = np.asarray(y_errs, dtype=float)
        jitter_half_width = min(0.38, 0.015 * len(y_vals))
        if len(y_vals) == 1:
            x = np.array([group_to_x[grp]], dtype=float)
        else:
            x = group_to_x[grp] + np.linspace(-jitter_half_width, jitter_half_width, len(y_vals))

        ax.errorbar(
            x,
            y_vals,
            yerr=np.where(np.isfinite(y_errs), y_errs, np.nan),
            fmt="o",
            color=group_colors[grp],
            ecolor=group_colors[grp],
            elinewidth=1.7,
            capsize=4,
            markersize=6,
            alpha=0.9,
            zorder=3,
        )

        stats = group_stats.get(grp)
        if stats:
            x_center = group_to_x[grp]
            half_span = 0.42
            ax.plot(
                [x_center - half_span, x_center + half_span],
                [stats["mode"], stats["mode"]],
                color=COLOR_YELLOW,
                linestyle="--",
                linewidth=1.6,
                alpha=0.95,
                zorder=2,
            )

    ax.set_xticks([group_to_x[g] for g in group_order])
    ax.set_xticklabels(group_order)
    ax.tick_params(axis="x", labelsize=26, which="both", length=0)
    ax.set_ylabel(r"Estimated $\Omega$ (mm)")
    style_axes(ax)

    y0, y1 = ax.get_ylim()
    y_text = y1 - 0.06 * (y1 - y0)
    for grp in group_order:
        stats = group_stats.get(grp)
        if not stats:
            continue
        txt = f"$n=${stats['n']}\n     mode$=${stats['mode']:.2f} mm"
        ax.text(
            group_to_x[grp] - 0.1,
            y_text,
            txt,
            color=group_colors[grp],
            ha="center",
            va="top",
            fontsize=22,
            family="monospace",
        )

    fig.savefig(out_svg, format="svg", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def gaussian_kernel_pdf(values, errors, x_grid, sigma_floor=1e-3):
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


def plot_kernel_distribution(rows, out_svg, group_stats):
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
        [float(r["wavelength_firstpeak_err_mm"]) for r in rows if "AgarTilted" in r["image"] and np.isfinite(r["wavelength_firstpeak_mm"])],
        dtype=float,
    )
    y_nontilted = np.array(
        [r["wavelength_firstpeak_mm"] for r in rows if "AgarTilted" not in r["image"] and np.isfinite(r["wavelength_firstpeak_mm"])],
        dtype=float,
    )
    e_nontilted = np.array(
        [float(r["wavelength_firstpeak_err_mm"]) for r in rows if "AgarTilted" not in r["image"] and np.isfinite(r["wavelength_firstpeak_mm"])],
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

        pdf = gaussian_kernel_pdf(yy, ee, x_grid=x_dense, sigma_floor=sigma_floor)
        ax.fill_between(x_dense, 0.0, pdf, color=color, alpha=0.22, linewidth=0)
        ax.plot(x_dense, pdf, color=color, linewidth=2.4, alpha=0.95)

        stats = group_stats.get(title)
        if stats:
            mode = stats["mode"]
            ci_low = stats["ci_low"]
            ci_high = stats["ci_high"]
            y_bar = float(np.nanmax(pdf) * 1.10 if np.isfinite(np.nanmax(pdf)) and np.nanmax(pdf) > 0 else 1.0)
            ax.axvline(mode, color=COLOR_YELLOW, linestyle="--", linewidth=1.6, alpha=1.0)
            ax.errorbar(
                mode,
                y_bar,
                xerr=np.array([[mode - ci_low], [ci_high - mode]]),
                fmt="none",
                ecolor=COLOR_YELLOW,
                elinewidth=2.0,
                capsize=6,
                capthick=2.0,
                zorder=5,
            )

        ax.set_ylabel("KDE density")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        style_axes(ax)

    axes[-1].set_xlabel(r"Estimated $\Omega$ (mm)")

    fig.savefig(out_svg, format="svg", transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def main():
    set_plot_style()
    input_dir = Path("data/col_segmented/edited")
    output_dir = Path(__file__).resolve().parent / "plots" / "mode_errorbars_kde"

    dx_pixels = 8
    min_root_length = 200
    max_lag = 200
    peak_threshold = 0.0
    n_bootstrap = 2000
    ci_percent = 95.0

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

    groups = split_groups(filtered_rows)
    group_stats = {}
    for group_name, group_rows in groups.items():
        wavelengths = wavelengths_from_rows(group_rows)
        errors = errors_from_rows(group_rows)
        if wavelengths.size == 0:
            continue

        smooth_sigma = float(np.median(errors)) if errors.size > 0 else 0.0
        mode_result = bootstrap_mode_ci(
            wavelengths,
            n_bootstrap=n_bootstrap,
            ci=ci_percent,
            random_seed=42,
            smooth_sigma=smooth_sigma,
        )
        group_stats[group_name] = {
            "n": int(wavelengths.size),
            "mode": float(mode_result["mode"]),
            "ci_low": float(mode_result["ci_lower"]),
            "ci_high": float(mode_result["ci_upper"]),
        }

    errorbar_svg = output_dir / f"mode_errorbars_tilted_vs_nontilted_ct_{peak_threshold:.2f}.svg"
    kde_svg = output_dir / f"mode_histogram_kde_tilted_vs_nontilted_filtered_ct_{peak_threshold:.2f}.svg"

    plot_errorbars(filtered_rows, errorbar_svg, group_stats)
    plot_kernel_distribution(filtered_rows, kde_svg, group_stats)

    print(f"\nTotal root estimates after peak filter (> {peak_threshold}): {len(filtered_rows)}")
    print(f"Errorbar plot saved to {errorbar_svg}")
    print(f"Histogram-kernel plot saved to {kde_svg}")


if __name__ == "__main__":
    main()