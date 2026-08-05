"""How far a root grows per oscillation, and how that changes as it ages.

Counts the waves along every segmented root (see arclength_events.py) and plots
the arc length s of each successive baseline crossing against the oscillation
index n.  Under the assumptions set out there -- one fixed period T for all
roots, an elongation rate v that is shared but grows as the seedling ages -- the
mm per oscillation Omega = v*T is a measurement of v, resolved along the root.

  1. s(n) for every root, shifted to s(0) = 0, tilted plates beside vertical
     ones, against the straight line through the ends of the group median.  The
     median bowing below its own chord is Omega growing as the root grows.  This
     curve is a running sum and so is smooth whatever the data do: read it for
     the direction of the effect, not as evidence of its strength.
  2. the local wavelength s(n+1) - s(n) against n -- the same information
     differentiated, and the honest picture of how noisy a single root is -- in
     mm and rescaled by each root's own mean.
  3. the late wavelength of each root against its own early wavelength.  Both
     curves above aggregate across roots at fixed n, where the roots still
     present at large n are the short-wavelength ones; this comparison stays
     inside one root and so is free of that.  It is the result.
  4. what it costs the wavelength estimate: Omega measured for a whole root
     against the length of that root.  A number quoted for a plate is an average
     over however much root was in the picture, so plates imaged later report a
     longer wavelength -- the trend of figure 3 seen across roots instead of
     within them.
  5. robustness: the late/early ratio against the hysteresis threshold and the
     baseline window, beside the value from a sliding-window Fourier estimate
     that has no threshold in it at all.
  6. a diagnostic: detrended theta(s) for roots of the 1.5% agar pair, tilted and
     vertical, with the detected crossings marked, so the counting can be checked
     by eye.

Aggregate curves are drawn bold while at least half of the group still has data
and thin once fewer do, with an interquartile band throughout.

Every figure is produced twice.  Across roots the wavelengths are not symmetric
-- a handful of roots sit far above the bulk and none can sit below zero -- so
where the figures collapse many roots into one number they do it once with the
median, into plots/medians/, and once with the half-sample mode, into plots/mode/.  The
mode is the estimator ../wavelengths settled on for exactly this reason, so the
two folders can be compared with the autocorrelation results directly.

It matters most for the gap between the two groups.  By median the vertical roots
look 1.8 mm longer in Omega than the tilted ones (6.0 against 4.3); by mode the
gap halves (5.4 against 4.2), because it is the tail of long-wavelength vertical
roots, not the bulk of them, that is doing the separating.  Everything else --
the rise of Omega along a root, its size, its robustness -- is unchanged by the
switch.

Only aggregates across roots change between the folders.  Anything computed
inside a single root stays a median in both, because half a root holds five or
ten intervals and a mode of that is meaningless.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from arclength_events import (
    COLOR_NON_TILTED,
    COLOR_REFERENCE,
    COLOR_TILTED,
    COLOR_YELLOW,
    aligned_curves,
    analyze_image,
    central,
    central_by_column,
    detect_crossings,
    detrend_theta,
    early_late_wavelengths,
    local_wavelengths,
    set_plot_style,
    split_groups,
    stack_on_index_grid,
    style_axes,
    windowed_wavelength,
)


GROUP_ORDER = ["Tilted", "Vertical"]
GROUP_COLORS = {"Tilted": COLOR_TILTED, "Vertical": COLOR_NON_TILTED}
MIN_ROOTS = 3  # stop drawing the centre line once fewer roots than this are left
CENTRE_LABEL = {"median": "median", "mode": "mode"}
DIAGNOSTIC_IMAGES = {
    "Tilted": "Col-0_20220630112924-0001_1.5AgarTilted75.png",
    "Vertical": "Col-0_20220630112924-0001_1.5AgarVertical.png",
}


def save_figure(fig, out_path):
    """Write the figure in the format its extension asks for, on a clear ground."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        format=out_path.suffix.lstrip(".") or "svg",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)


def outlined(line):
    """White halo so a bold median stays readable over its own bundle of roots."""
    line.set_path_effects(
        [path_effects.Stroke(linewidth=line.get_linewidth() + 1.8, foreground="white"), path_effects.Normal()]
    )


def draw_centre(ax, grid, values, color, centre="median", label=None):
    """Centre line and interquartile band, drawn thin where the group has thinned.

    Both masks are prefixes of the grid, since a root that has run out of length
    never comes back, so the thin and bold pieces join without a gap.  The band
    stays the quartiles whichever centre is chosen: it is a description of the
    spread, not a competing estimate of the middle.
    """
    if values.size == 0:
        return
    counts = np.sum(np.isfinite(values), axis=0)
    keep = counts >= MIN_ROOTS
    solid = counts >= max(MIN_ROOTS, int(np.ceil(0.5 * values.shape[0])))
    if not keep.any():
        return

    mid = central_by_column(values, centre)
    with np.errstate(invalid="ignore"):
        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)

    ax.fill_between(grid[keep], q1[keep], q3[keep], color=color, alpha=0.30, linewidth=0, zorder=3)
    (thin,) = ax.plot(grid[keep], mid[keep], color=color, linewidth=1.8, zorder=4)
    outlined(thin)
    if solid.any():
        (line,) = ax.plot(grid[solid], mid[solid], color=color, linewidth=4.0, zorder=5, label=label)
        outlined(line)
    return mid, keep, solid


def plot_arclength_vs_index(rows, out_svg, centre="median"):
    groups = split_groups(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.8), sharey=True, constrained_layout=True)
    fig.patch.set_alpha(0.0)

    y_max = 0.0
    for ax, grp in zip(axes, GROUP_ORDER):
        curves = aligned_curves(groups[grp])
        color = GROUP_COLORS[grp]
        if not curves:
            continue

        # Individual roots in grey, so the group median can be read off in colour.
        for index, s_rel in curves:
            ax.plot(index, s_rel, color=COLOR_REFERENCE, alpha=0.40, linewidth=1.0, zorder=2)
            y_max = max(y_max, float(np.max(s_rel)))

        grid, values = stack_on_index_grid(curves)
        drawn = draw_centre(ax, grid, values, color, centre=centre)
        if drawn is not None:
            mid, keep, solid = drawn
            # Constant growth per oscillation, anchored on the well-sampled stretch.
            ends = np.where(solid)[0][[0, -1]]
            ax.plot(
                grid[ends],
                mid[ends],
                color="black",
                linestyle="--",
                linewidth=1.8,
                zorder=6,
                dashes=(5, 4),
            )

        ax.set_xlabel("Oscillation index $n$")
        ax.set_xlim(left=0)
        ax.text(
            0.04,
            0.95,
            f"{grp}\n$N={len(curves)}$ roots",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=24,
        )
        style_axes(ax)
        ax.grid(True, axis="x", which="major", alpha=0.3, linewidth=0.7)

    axes[0].set_ylim(0, 1.04 * y_max)  # shared, so it has to cover both panels
    axes[0].set_ylabel(r"Arc length $s(n) - s(0)$ (mm)")
    save_figure(fig, out_svg)


def local_wavelength_matrix(rows):
    """Local Omega of every root of a group, on the shared half-integer index grid."""
    curves = [local_wavelengths(row) for row in rows]
    curves = [(idx, om) for idx, om in curves if idx.size]
    if not curves:
        return np.array([]), np.zeros((0, 0))
    n_max = max(len(idx) for idx, _ in curves)
    values = np.full((len(curves), n_max), np.nan, dtype=float)
    for i, (_, omega) in enumerate(curves):
        values[i, : len(omega)] = omega
    grid = 0.5 * np.arange(n_max, dtype=float) + 0.5
    return grid, values


def plot_local_wavelength(rows, out_svg, centre="median"):
    groups = split_groups(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.8), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    for grp in GROUP_ORDER:
        color = GROUP_COLORS[grp]
        grid, values = local_wavelength_matrix(groups[grp])
        if grid.size == 0:
            continue

        # Left: mm as measured.  Right: each root divided by its own middle value,
        # which takes out the root-to-root spread and leaves only the shape.  That
        # divisor is a median even in the mode figures: one root has ten or twenty
        # local wavelengths in it, nowhere near enough to find a mode.
        per_root = np.array([[central(row_vals, "median")] for row_vals in values])
        scaled = values / per_root

        for ax, data in zip(axes, (values, scaled)):
            for row_vals in data:
                ax.plot(grid, row_vals, color=color, alpha=0.13, linewidth=1.0, zorder=2)
            draw_centre(ax, grid, data, color, centre=centre, label=f"{grp} ($N={data.shape[0]}$)")

    axes[0].set_ylabel(r"Local $\Omega$ (mm/oscillation)")
    axes[0].set_ylim(0, 16)
    axes[1].set_ylabel(r"Local $\Omega$ / root median")
    axes[1].set_ylim(0, 2.2)
    axes[1].axhline(1.0, color=COLOR_REFERENCE, linewidth=1.4, zorder=1)

    for ax in axes:
        ax.set_xlabel("Oscillation index $n$")
        ax.set_xlim(left=0)
        style_axes(ax)
        ax.grid(True, axis="x", which="major", alpha=0.3, linewidth=0.7)
    axes[0].legend(loc="upper left", frameon=False)

    save_figure(fig, out_svg)


def early_late_pairs(rows):
    pairs = np.array([early_late_wavelengths(r["crossings_mm"]) for r in rows], dtype=float)
    if pairs.size == 0:
        return np.array([]), np.array([])
    pairs = pairs[np.all(np.isfinite(pairs), axis=1)]
    if pairs.size == 0:
        return np.array([]), np.array([])
    return pairs[:, 0], pairs[:, 1]


def plot_early_vs_late(rows, out_svg, centre="median"):
    groups = split_groups(rows)

    fig, ax = plt.subplots(figsize=(7.6, 7.2), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    summary = {}
    all_vals = []
    for grp in GROUP_ORDER:
        early, late = early_late_pairs(groups[grp])
        if early.size == 0:
            continue
        all_vals.extend([early, late])

        ax.plot(
            early,
            late,
            "o",
            color=GROUP_COLORS[grp],
            markersize=9,
            alpha=0.85,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
            label=f"{grp} ($N={len(early)}$)",
        )
        summary[grp] = {
            "n": int(len(early)),
            "centre": centre,
            "ratio": central(late / early, centre),
            "fraction_late_longer": float(np.mean(late > early)),
            "wilcoxon_p": float(stats.wilcoxon(late, early).pvalue) if len(early) >= 6 else np.nan,
        }

    if all_vals:
        flat = np.concatenate(all_vals)
        lo, hi = 0.85 * float(np.min(flat)), 1.06 * float(np.max(flat))
        ax.plot([lo, hi], [lo, hi], color=COLOR_REFERENCE, linestyle="--", linewidth=1.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    lines = [
        f"{grp}: {st['fraction_late_longer'] * 100:.0f}% above, "
        f"$\\times${st['ratio']:.2f} ({CENTRE_LABEL[centre]}), $p=${st['wilcoxon_p']:.0e}"
        for grp, st in summary.items()
    ]
    if lines:
        ax.text(0.97, 0.06, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=17)

    ax.set_xlabel(r"Early $\Omega$ (mm)")
    ax.set_ylabel(r"Late $\Omega$ (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False)
    style_axes(ax)
    ax.grid(True, axis="x", which="major", alpha=0.3, linewidth=0.7)

    save_figure(fig, out_svg)
    return summary


def plot_wavelength_vs_length(rows, out_svg, centre="median"):
    """Whole-root Omega against root length: the same drift seen across roots.

    Length stands in for age, so this mixes the ages of the plates with whatever
    else differs between them.  It is not only a between-plate effect, though:
    the correlation is positive inside every one of the eight images taken
    separately, rho from +0.24 to +0.90.
    """
    groups = split_groups(rows)

    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    summary = {}
    for grp in GROUP_ORDER:
        grp_rows = groups[grp]
        length = np.array([r["main_length_mm"] for r in grp_rows], dtype=float)
        omega = np.array([r["slope_mm_per_osc"] for r in grp_rows], dtype=float)
        err = np.array([r["slope_err_mm_per_osc"] for r in grp_rows], dtype=float)
        if length.size == 0:
            continue

        ax.errorbar(
            length,
            omega,
            yerr=np.where(np.isfinite(err) & (err > 0), err, np.nan),
            fmt="o",
            color=GROUP_COLORS[grp],
            ecolor=GROUP_COLORS[grp],
            elinewidth=1.3,
            capsize=3,
            markersize=8,
            alpha=0.85,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
            label=f"{grp} ($N={length.size}$)",
        )

        rho, p_value = stats.spearmanr(length, omega)
        summary[grp] = {
            "n": int(length.size),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "centre": centre,
            "omega_centre_mm": central(omega, centre),
        }
        if length.size >= 4:
            span = np.array([length.min(), length.max()])
            if centre == "mode":
                # Theil-Sen: the median of the pairwise slopes, so the few roots
                # far above the bulk cannot tilt the line the way least squares
                # lets them.
                slope, intercept, _, _ = stats.theilslopes(omega, length)
                trend = intercept + slope * span
            else:
                trend = np.polyval(np.polyfit(length, omega, 1), span)
            ax.plot(span, trend, color=GROUP_COLORS[grp], linewidth=2.0, alpha=0.55, zorder=2)

    lines = [
        f"{grp}: $\\rho=${st['spearman_rho']:+.2f}, {CENTRE_LABEL[centre]} $\\Omega=${st['omega_centre_mm']:.2f} mm"
        for grp, st in summary.items()
    ]
    if lines:
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=17)

    ax.set_xlabel("Root length (mm)")
    ax.set_ylabel(r"$\Omega$ of the whole root (mm)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", frameon=False)
    style_axes(ax)
    ax.grid(True, axis="x", which="major", alpha=0.3, linewidth=0.7)

    save_figure(fig, out_svg)
    return summary


def _ratio_for_settings(raw_roots, smooth_mm, baseline_mm, rel_threshold, centre="median"):
    """Late/early ratio per group when the detection is set up this way."""
    out = {}
    for grp in GROUP_ORDER:
        ratios = []
        for root in raw_roots:
            if root["group"] != grp:
                continue
            detrended = detrend_theta(root["s_mm"], root["theta"], smooth_mm=smooth_mm, baseline_mm=baseline_mm)
            crossings, _ = detect_crossings(root["s_mm"], detrended, rel_threshold=rel_threshold)
            early, late = early_late_wavelengths(crossings)
            if np.isfinite(early) and np.isfinite(late) and early > 0:
                ratios.append(late / early)
        out[grp] = central(ratios, centre) if ratios else np.nan
    return out


def _windowed_ratio(raw_roots, centre="median"):
    """The same ratio from the sliding-window Fourier estimate: no threshold at all."""
    out = {}
    for grp in GROUP_ORDER:
        ratios = []
        for root in raw_roots:
            if root["group"] != grp:
                continue
            _, lam = windowed_wavelength(root["s_mm"], root["theta"])
            if len(lam) < 6:
                continue
            half = len(lam) // 2
            ratios.append(np.median(lam[-half:]) / np.median(lam[:half]))
        out[grp] = central(ratios, centre) if ratios else np.nan
    return out


def plot_robustness(raw_roots, out_svg, smooth_mm, baseline_mm, rel_threshold, centre="median"):
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    baselines = [8.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    reference = _windowed_ratio(raw_roots, centre)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharey=True, constrained_layout=True)
    fig.patch.set_alpha(0.0)

    sweeps = (
        (axes[0], thresholds, [_ratio_for_settings(raw_roots, smooth_mm, baseline_mm, t, centre) for t in thresholds],
         "Hysteresis threshold $h/A$", rel_threshold),
        (axes[1], baselines, [_ratio_for_settings(raw_roots, smooth_mm, b, rel_threshold, centre) for b in baselines],
         "Baseline window (mm)", baseline_mm),
    )

    for ax, xs, results, xlabel, default in sweeps:
        for grp in GROUP_ORDER:
            color = GROUP_COLORS[grp]
            ax.plot(xs, [r[grp] for r in results], "o-", color=color, linewidth=2.6, markersize=9,
                    label=f"{grp}", zorder=3)
            if np.isfinite(reference[grp]):
                ax.axhline(reference[grp], color=color, linestyle=":", linewidth=2.0, alpha=0.9, zorder=2)
        ax.axhline(1.0, color=COLOR_REFERENCE, linewidth=1.6, zorder=1)
        ax.axvline(default, color=COLOR_YELLOW, linestyle="--", linewidth=1.8, alpha=0.9, zorder=1)
        ax.set_xlabel(xlabel)
        style_axes(ax)
        ax.grid(True, axis="x", which="major", alpha=0.3, linewidth=0.7)

    axes[0].set_ylabel(rf"Late $\Omega$ / early $\Omega$ ({CENTRE_LABEL[centre]})")
    axes[0].set_ylim(0.9, 1.8)
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].text(
        0.97,
        0.06,
        "dotted: windowed Fourier,\nno threshold",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=17,
    )

    save_figure(fig, out_svg)
    return {"windowed_fourier_ratio": reference}


def plot_theta_diagnostics(rows, out_svg, n_per_group=2):
    """Traces from the 1.5% agar pair -- same plate and day, tilted and vertical."""
    picked = []
    for grp in GROUP_ORDER:
        candidates = [r for r in rows if r["image"] == DIAGNOSTIC_IMAGES[grp]]
        candidates.sort(key=lambda r: r["root_label"])
        picked.extend((grp, row) for row in candidates[:n_per_group])
    if not picked:
        return

    fig, axes = plt.subplots(len(picked), 1, figsize=(9.5, 2.4 * len(picked)), constrained_layout=True)
    fig.patch.set_alpha(0.0)
    axes = np.atleast_1d(axes)

    for ax, (grp, row) in zip(axes, picked):
        color = GROUP_COLORS[grp]
        s_mm = np.asarray(row["s_mm"], dtype=float)
        detrended = np.asarray(row["theta_detrended"], dtype=float)
        crossings = np.asarray(row["crossings_mm"], dtype=float)

        ax.axhline(0.0, color=COLOR_REFERENCE, linewidth=1.2, zorder=1)
        ax.plot(s_mm, detrended, color=color, linewidth=1.8, zorder=2)
        ax.plot(
            crossings,
            np.zeros_like(crossings),
            "o",
            color=COLOR_YELLOW,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.set_ylabel(r"$\delta\theta$", fontsize=24)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.32 * (hi - lo))  # headroom for the caption
        ax.text(
            0.995,
            0.94,
            f"{grp}, {row['n_oscillations']:.1f} osc, $\\Omega=${row['slope_mm_per_osc']:.2f} mm",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=color,
            fontsize=17,
        )
        ax.tick_params(axis="both", labelsize=17)
        style_axes(ax)

    axes[-1].set_xlabel("Arc length $s$ (mm)", fontsize=24)
    save_figure(fig, out_svg)


def save_results(rows, out_json, params=None, summary=None):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    keep = (
        "image",
        "root_label",
        "mm_per_pixel",
        "dx_mm",
        "main_length_mm",
        "crossings_mm",
        "crossing_dirs",
        "oscillation_index",
        "n_oscillations",
        "amplitude_rad",
        "omega_early_mm",
        "omega_late_mm",
        "slope_mm_per_osc",
        "slope_err_mm_per_osc",
        "rms_residual_mm",
    )
    payload = {
        "params": params or {},
        "summary": summary or {},
        "roots": [{k: row[k] for k in keep} for row in rows],
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)


def main():
    set_plot_style()
    input_dir = Path("data/col_segmented/edited")
    here = Path(__file__).resolve().parent
    output_dir = here / "plots"

    dx_pixels = 8
    min_root_length = 200  # pixels; also where theta(s) starts, skipping the seedling
    smooth_mm = 1.0
    baseline_mm = 15.0
    rel_threshold = 0.3
    min_oscillations = 2.0

    files = sorted(input_dir.glob("*.png"))
    if not files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(files)} images to process\n")
    all_rows = []
    for fp in files:
        all_rows.extend(
            analyze_image(
                fp,
                dx_pixels=dx_pixels,
                min_root_length=min_root_length,
                smooth_mm=smooth_mm,
                baseline_mm=baseline_mm,
                rel_threshold=rel_threshold,
            )
        )

    rows = [r for r in all_rows if r["n_oscillations"] >= min_oscillations]
    if not rows:
        print("No roots survived the oscillation-count filter.")
        return

    # theta(s) alone, for the sweeps that have to redo the detection.
    raw_roots = [
        {"group": grp, "s_mm": np.asarray(r["s_mm"]), "theta": np.asarray(r["theta"])}
        for grp in GROUP_ORDER
        for r in split_groups(rows)[grp]
    ]

    tag = f"dx{dx_pixels}_base{baseline_mm:.0f}mm_thr{rel_threshold:.2f}"
    results_json = here / "outputs" / f"arclength_rows_{tag}.json"

    # The same figures twice: once with medians across roots, once with the
    # half-sample mode of ../wavelengths, which the tail of long-wavelength roots
    # cannot drag upwards.
    summary = {}
    for centre, folder in (("median", output_dir / "medians"), ("mode", output_dir / "mode")):
        plot_arclength_vs_index(rows, folder / f"arclength_vs_index_{tag}.svg", centre=centre)
        plot_local_wavelength(rows, folder / f"local_wavelength_vs_index_{tag}.svg", centre=centre)
        early_late = plot_early_vs_late(rows, folder / f"early_vs_late_wavelength_{tag}.svg", centre=centre)
        length_trend = plot_wavelength_vs_length(
            rows, folder / f"wavelength_vs_root_length_{tag}.svg", centre=centre
        )
        robustness = plot_robustness(
            raw_roots,
            folder / f"detection_robustness_{tag}.svg",
            smooth_mm,
            baseline_mm,
            rel_threshold,
            centre=centre,
        )
        # No aggregate in the diagnostic, but each folder should stand alone.
        plot_theta_diagnostics(rows, folder / f"theta_crossings_examples_{tag}.svg")

        summary[centre] = {
            "early_vs_late": early_late,
            "omega_vs_length": length_trend,
            "robustness": robustness,
        }
        print(f"\nWrote the {centre} figures to {folder}")
    save_results(
        rows,
        results_json,
        params={
            "dx_pixels": dx_pixels,
            "min_root_length_px": min_root_length,
            "smooth_mm": smooth_mm,
            "baseline_mm": baseline_mm,
            "rel_threshold": rel_threshold,
            "min_oscillations": min_oscillations,
        },
        summary=summary,
    )

    groups = split_groups(rows)
    print(f"\nRoots with >= {min_oscillations} oscillations: {len(rows)} of {len(all_rows)}")
    for centre in ("median", "mode"):
        print(f"\n  across roots by {centre}:")
        for grp in GROUP_ORDER:
            grp_rows = groups[grp]
            if not grp_rows:
                continue
            slopes = [r["slope_mm_per_osc"] for r in grp_rows]
            n_osc = [r["n_oscillations"] for r in grp_rows]
            el = summary[centre]["early_vs_late"].get(grp, {})
            lt = summary[centre]["omega_vs_length"].get(grp, {})
            print(
                f"    {grp:9s} N={len(grp_rows):3d}"
                f"  Omega={central(slopes, centre):5.2f} mm"
                f"  oscillations={central(n_osc, centre):4.1f}"
                f"  late/early={el.get('ratio', np.nan):.2f}"
                f" ({el.get('fraction_late_longer', np.nan):.2f} longer, p={el.get('wilcoxon_p', np.nan):.1e})"
                f"  Omega vs length rho={lt.get('spearman_rho', np.nan):+.2f}"
            )
        print(f"    windowed-Fourier late/early: {summary[centre]['robustness']['windowed_fourier_ratio']}")
    print(f"\nSaved {results_json}")


if __name__ == "__main__":
    main()
