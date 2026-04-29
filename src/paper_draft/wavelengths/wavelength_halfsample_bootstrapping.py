"""Half-sample mode + bootstrap uncertainty analysis for wavelength data.

Method summary aligned with the requested rationale:
1) Half-sample mode (HSM) is used as a robust modal estimator for unimodal,
   heavy-tailed samples.
2) Uncertainty is quantified by nonparametric bootstrap: recompute HSM on each
   resample, then use percentile bounds (e.g., 2.5 and 97.5) as an empirical
   interval.
3) Bootstrap intervals are interpreted as a descriptive uncertainty measure,
   not a strict asymptotic CI for density modes.

References (for manuscript methods text):
- Bickel DR, Fruehwirth R (2006)
- Hedges SB, Shah P (2003)
- Romano JP (1988)

Input JSON format:
- List[dict] rows containing at least:
  - image
  - wavelength_firstpeak_mm
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from errorbars_filter import analyze_image, filter_rows_by_peak_threshold, set_plot_style, style_axes


def _hsm_recursive(sorted_vals):
    """Recursive half-sample mode on sorted 1D data."""
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
    """Compute half-sample mode (HSM) as a robust mode estimator."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(_hsm_recursive(np.sort(x)))


def bootstrap_mode_ci(data, n_bootstrap=2000, ci=95.0, random_seed=42, smooth_sigma=0.0):
    """Bootstrap percentile interval for the HSM.

    Returns an empirical interval from bootstrap quantiles; use as descriptive
    uncertainty for mode estimates.
    """
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
            # Smoothed bootstrap: avoids degenerate intervals for discrete/small-n data.
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


def compare_with_growth_speed_estimate(mode_result, wavelength_from_vg, vg_uncertainty=None):
    """Compare v_g-based wavelength estimate against HSM bootstrap interval."""
    ci_lower = float(mode_result["ci_lower"])
    ci_upper = float(mode_result["ci_upper"])
    mode = float(mode_result["mode"])
    wv = float(wavelength_from_vg)

    within_ci = ci_lower <= wv <= ci_upper
    distance_to_ci = 0.0 if within_ci else float(min(abs(wv - ci_lower), abs(wv - ci_upper)))
    rel_err = np.nan if (not np.isfinite(mode) or mode == 0.0) else float(abs(wv - mode) / abs(mode) * 100.0)

    return {
        "wavelength_from_vg": wv,
        "mode": mode,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "within_ci": bool(within_ci),
        "distance_to_ci": float(distance_to_ci),
        "relative_error_percent": rel_err,
        "vg_uncertainty": None if vg_uncertainty is None else float(vg_uncertainty),
    }


def load_rows(json_path):
    """Load rows exported from errorbars_filter.py."""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of row dictionaries.")
    return rows


def build_rows_from_images(
    input_dir,
    dx_pixels=8,
    min_root_length=200,
    max_lag=200,
    peak_threshold=0.1,
):
    """Generate wavelength rows directly from segmented PNG images."""
    files = sorted(Path(input_dir).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    set_plot_style()
    all_rows = []
    print(f"Found {len(files)} images to process")
    for fp in files:
        rows = analyze_image(
            fp,
            dx_pixels=dx_pixels,
            min_root_length=min_root_length,
            max_lag=max_lag,
        )
        all_rows.extend(rows)

    filtered_rows = filter_rows_by_peak_threshold(all_rows, peak_threshold=peak_threshold)
    print(f"Total root estimates after peak filter ({peak_threshold}): {len(filtered_rows)}")
    return filtered_rows


def split_groups(rows):
    """Split rows into tilted and non-tilted groups by image name."""
    groups = {
        "Tilted": [r for r in rows if "AgarTilted" in str(r.get("image", ""))],
        "Non-tilted": [r for r in rows if "AgarTilted" not in str(r.get("image", ""))],
    }
    return groups


def wavelengths_from_rows(rows, key="wavelength_firstpeak_mm"):
    """Extract finite wavelengths from row dictionaries."""
    vals = []
    for r in rows:
        v = float(r.get(key, np.nan))
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def errors_from_rows(rows, key="wavelength_firstpeak_err_mm"):
    """Extract finite positive per-root wavelength errors, if available."""
    vals = []
    for r in rows:
        v = float(r.get(key, np.nan))
        if np.isfinite(v) and v > 0:
            vals.append(v)
    return np.asarray(vals, dtype=float)


def plot_group_result(group_name, wavelengths, mode_result, out_path, vg_compare=None, x_limits=None):
    """Plot histogram and bootstrap mode distribution for one group."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    fig.patch.set_alpha(0.0)

    bins = max(6, min(25, len(wavelengths) // 2 if len(wavelengths) > 0 else 6))
    ax1.hist(wavelengths, bins=bins, alpha=0.75, edgecolor="black", color="#5B8DB8")

    mode = mode_result["mode"]
    lo = mode_result["ci_lower"]
    hi = mode_result["ci_upper"]
    ax1.axvline(mode, color="#B22222", linestyle="--", linewidth=2.0, label=f"HSM = {mode:.2f} mm")
    ax1.axvline(lo, color="#E38B29", linestyle=":", linewidth=2.0)
    ax1.axvline(hi, color="#E38B29", linestyle=":", linewidth=2.0, label=f"{mode_result['ci_percent']:.0f}% interval")
    ax1.fill_betweenx(ax1.get_ylim(), lo, hi, color="#E38B29", alpha=0.2)

    if vg_compare is not None and np.isfinite(vg_compare["wavelength_from_vg"]):
        color = "#2E8B57" if vg_compare["within_ci"] else "#B22222"
        ax1.axvline(
            vg_compare["wavelength_from_vg"],
            color=color,
            linestyle="-.",
            linewidth=2.0,
            label=f"v_g estimate = {vg_compare['wavelength_from_vg']:.2f} mm",
        )

    ax1.set_title(f"{group_name}: $\Omega$ Distribution (n={len(wavelengths)})")
    ax1.set_xlabel("Arc-length $\Omega$ (mm)")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="both", labelsize=16)
    style_axes(ax1)
    ax1.legend(fontsize=12, loc="upper right")

    bs = mode_result["bootstrap_modes"]
    ax2.hist(bs, bins=30, alpha=0.75, edgecolor="black", color="#5B8DB8")
    ax2.axvline(mode, color="#B22222", linestyle="--", linewidth=2.0, label="HSM (original sample)")
    ax2.axvline(lo, color="#E38B29", linestyle=":", linewidth=2.0)
    ax2.axvline(hi, color="#E38B29", linestyle=":", linewidth=2.0, label="Empirical interval bounds")
    ax2.set_title(f"{group_name}: Bootstrap Mode Distribution")
    ax2.set_xlabel("Bootstrap HSM (mm)")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="both", labelsize=16)
    style_axes(ax2)
    ax2.legend(fontsize=12, loc="upper right")

    # Use shared x-axis limits when provided so tilted and non-tilted figures match.
    if x_limits is None:
        x_min = min(np.min(wavelengths), np.min(bs))
        x_max = max(np.max(wavelengths), np.max(bs))
        margin = 0.05 * (x_max - x_min)
        x_min -= margin
        x_max += margin
    else:
        x_min, x_max = x_limits
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def run_analysis(rows, out_dir, n_bootstrap, ci, vg_tilted=None, vg_nontilted=None, use_smoothed_bootstrap=True):
    """Run HSM + bootstrap analysis for tilted/non-tilted groups."""
    groups = split_groups(rows)
    results = {}
    plot_inputs = {}

    for group_name, group_rows in groups.items():
        wl = wavelengths_from_rows(group_rows)
        wl_err = errors_from_rows(group_rows)
        if wl.size < 5:
            print(f"[WARN] {group_name}: insufficient data (n={wl.size}), skipping.")
            continue

        smooth_sigma = float(np.median(wl_err)) if (use_smoothed_bootstrap and wl_err.size > 0) else 0.0
        mode_result = bootstrap_mode_ci(
            wl,
            n_bootstrap=n_bootstrap,
            ci=ci,
            random_seed=42,
            smooth_sigma=smooth_sigma,
        )
        bs = mode_result["bootstrap_modes"]
        bs_unique = int(np.unique(np.round(bs, 6)).size)
        print(f"\n{group_name}")
        print(f"  n = {wl.size}")
        print(f"  mean = {np.mean(wl):.3f} mm")
        print(f"  median = {np.median(wl):.3f} mm")
        print(f"  HSM = {mode_result['mode']:.3f} mm")
        print(f"  empirical {ci:.0f}% interval = [{mode_result['ci_lower']:.3f}, {mode_result['ci_upper']:.3f}] mm")
        print(f"  bootstrap std = {np.std(bs):.3f} mm, unique modes = {bs_unique}")
        if smooth_sigma > 0.0:
            print(f"  smoothed bootstrap sigma = {smooth_sigma:.3f} mm")

        vg_cmp = None
        if group_name == "Tilted" and vg_tilted is not None:
            vg_cmp = compare_with_growth_speed_estimate(mode_result, vg_tilted)
        if group_name == "Non-tilted" and vg_nontilted is not None:
            vg_cmp = compare_with_growth_speed_estimate(mode_result, vg_nontilted)

        if vg_cmp is not None:
            print(f"  v_g estimate = {vg_cmp['wavelength_from_vg']:.3f} mm")
            print(f"  within interval = {vg_cmp['within_ci']}")
            if not vg_cmp["within_ci"]:
                print(f"  distance to interval = {vg_cmp['distance_to_ci']:.3f} mm")

        plot_inputs[group_name] = {
            "wavelengths": wl,
            "mode_result": mode_result,
            "vg_compare": vg_cmp,
        }

        results[group_name] = {
            "n": int(wl.size),
            "mean": float(np.mean(wl)),
            "median": float(np.median(wl)),
            "mode_result": {
                "mode": float(mode_result["mode"]),
                "ci_lower": float(mode_result["ci_lower"]),
                "ci_upper": float(mode_result["ci_upper"]),
                "ci_percent": float(mode_result["ci_percent"]),
                "n_bootstrap": int(mode_result["n_bootstrap"]),
            },
            "vg_comparison": vg_cmp,
        }

    all_plot_values = []
    for item in plot_inputs.values():
        all_plot_values.append(item["wavelengths"])
        all_plot_values.append(item["mode_result"]["bootstrap_modes"])

    if all_plot_values:
        combined = np.concatenate([np.asarray(values, dtype=float) for values in all_plot_values if np.size(values) > 0])
        if combined.size > 0:
            x_min = float(np.min(combined))
            x_max = float(np.max(combined))
            span = x_max - x_min
            margin = 0.05 * span if span > 0.0 else 0.5
            shared_xlim = (x_min - margin, x_max + margin)
        else:
            shared_xlim = None
    else:
        shared_xlim = None

    for group_name, item in plot_inputs.items():
        plot_path = out_dir / f"hsm_bootstrap_{group_name.lower().replace(' ', '_')}.svg"
        plot_group_result(
            group_name,
            item["wavelengths"],
            item["mode_result"],
            plot_path,
            vg_compare=item["vg_compare"],
            x_limits=shared_xlim,
        )
        print(f"  saved plot: {plot_path}")

    return results


def main():
    # Edit these values directly for your dataset and theory comparison.
    project_root = Path(__file__).resolve().parents[3]
    input_json = Path(__file__).resolve().parent / "results" / "wavelength_rows.json"
    segmented_input_dir = project_root / "data" / "col_segmented" / "edited"
    out_dir = Path(__file__).resolve().parent / "plots" / "mode_bootstrap"
    n_bootstrap = 2000
    ci_percent = 99.0
    vg_tilted = None
    vg_nontilted = None
    use_smoothed_bootstrap = True

    # Parameters used only if JSON is missing and rows are built from images.
    dx_pixels = 8
    min_root_length = 200
    max_lag = 200
    peak_threshold = 0.0

    # Always apply paper_draft plotting defaults.
    set_plot_style()

    out_dir.mkdir(parents=True, exist_ok=True)

    if input_json.exists():
        rows = load_rows(input_json)
        print(f"Loaded rows from {input_json}")
    else:
        print(f"Input JSON not found at {input_json}")
        print(f"Building rows from images in {segmented_input_dir}")
        rows = build_rows_from_images(
            segmented_input_dir,
            dx_pixels=dx_pixels,
            min_root_length=min_root_length,
            max_lag=max_lag,
            peak_threshold=peak_threshold,
        )
        input_json.parent.mkdir(parents=True, exist_ok=True)
        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved generated rows to {input_json}")

    print("=" * 72)
    print("Half-sample mode with bootstrap empirical intervals")
    print("Interpretation: descriptive uncertainty for modal estimate")
    print("=" * 72)

    results = run_analysis(
        rows,
        out_dir=out_dir,
        n_bootstrap=n_bootstrap,
        ci=ci_percent,
        vg_tilted=vg_tilted,
        vg_nontilted=vg_nontilted,
        use_smoothed_bootstrap=use_smoothed_bootstrap,
    )

    summary_path = out_dir / "hsm_bootstrap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
