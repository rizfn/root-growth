from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autocorr_common import ensure_autocorr_cache


def is_tilted_image(name):
	return "AgarTilted" in name


def plot_errorbars_by_image(rows, out_png):
	"""Errorbar plot similar to estimate_wavelengths_col.py using first-peak only."""
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

	fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(image_names)), 6))

	legend_done = set()
	for img in image_names:
		pts = grouped[img]
		n = len(pts)
		jitter = np.array([0.0]) if n == 1 else np.linspace(-0.25, 0.25, n)

		yvals = np.array([p["wavelength_mm"] for p in pts if np.isfinite(p["wavelength_mm"])], dtype=float)
		lbl = f"{float(np.mean(yvals)):.2f} mm" if yvals.size > 0 else "n/a"

		for j, row in enumerate(pts):
			x = image_to_x[img] + jitter[j]
			color = image_to_color[img]
			label = lbl if img not in legend_done else None

			y = row["wavelength_mm"]
			yerr = row["wavelength_err_mm"]
			ax.errorbar(
				x,
				y,
				yerr=yerr if np.isfinite(yerr) else None,
				fmt="o",
				color=color,
				ecolor=color,
				elinewidth=1.7,
				capsize=4,
				markersize=6,
				alpha=0.9,
				zorder=3,
				label=label,
			)
			legend_done.add(img)

	ax.set_xticks(range(len(image_names)))
	ax.set_xticklabels(image_names, rotation=35, ha="right")
	ax.set_ylabel("Estimated wavelength (mm)")
	ax.set_xlabel("Image")
	ax.set_title("Per-root wavelength estimates from first autocorrelation peak")
	ax.grid(True, axis="y", alpha=0.25)
	if len(image_names) <= 20:
		ax.legend(fontsize=8, loc="best", framealpha=0.7)

	fig.tight_layout()
	fig.savefig(out_png, dpi=180, bbox_inches="tight")
	plt.close(fig)


def plot_histogram_combined(rows, out_png):
	"""Three vertical subplots (all/tilted/vertical) with shared bins and x-axis."""
	out_png.parent.mkdir(parents=True, exist_ok=True)
	y_all = np.array([r["wavelength_mm"] for r in rows if np.isfinite(r["wavelength_mm"])], dtype=float)
	y_tilted = np.array([r["wavelength_mm"] for r in rows if is_tilted_image(r["image"]) and np.isfinite(r["wavelength_mm"])], dtype=float)
	y_vertical = np.array([r["wavelength_mm"] for r in rows if (not is_tilted_image(r["image"])) and np.isfinite(r["wavelength_mm"])], dtype=float)

	if y_all.size == 0:
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.text(0.5, 0.5, "No valid wavelength estimates", ha="center", va="center")
		ax.set_axis_off()
		fig.tight_layout()
		fig.savefig(out_png, dpi=160, bbox_inches="tight")
		plt.close(fig)
		return

	p_lo = float(np.percentile(y_all, 2))
	p_hi = float(np.percentile(y_all, 98))
	if p_hi <= p_lo:
		center = float(np.median(y_all))
		spread = max(float(np.std(y_all)), 1e-3)
		p_lo, p_hi = center - 2.0 * spread, center + 2.0 * spread

	bins = np.linspace(p_lo, p_hi, 22)
	if np.allclose(bins[0], bins[-1]):
		bins = np.linspace(p_lo - 1.0, p_hi + 1.0, 22)

	fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
	datasets = [
		("All", y_all, "tab:blue"),
		("Tilted", y_tilted, "tab:orange"),
		("Vertical", y_vertical, "tab:green"),
	]

	for ax, (name, yy, color) in zip(axes, datasets):
		if yy.size > 0:
			ax.hist(yy, bins=bins, color=color, alpha=0.72, edgecolor="white", linewidth=0.8)
			mean_val = float(np.mean(yy))
			ax.axvline(mean_val, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
			ax.text(
				0.98,
				0.92,
				f"n={yy.size}, mean={mean_val:.2f} mm",
				transform=ax.transAxes,
				ha="right",
				va="top",
				fontsize=9,
				bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.75, "pad": 2},
			)
		else:
			ax.text(0.5, 0.5, "No valid roots", transform=ax.transAxes, ha="center", va="center")
		ax.set_title(f"{name} roots")
		ax.set_ylabel("Count")
		ax.grid(True, axis="y", alpha=0.25)

	axes[-1].set_xlabel("Estimated wavelength (mm)")
	fig.suptitle("Wavelength distributions by image type", y=0.995)
	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
	fig.savefig(out_png, dpi=180, bbox_inches="tight")
	plt.close(fig)


def plot_filtered_fraction(_kept_rows, summary_rows, out_png, method_label):
	"""Per-image pie charts (kept vs filtered) ordered like errorbar plot image order."""
	out_png.parent.mkdir(parents=True, exist_ok=True)

	if not summary_rows:
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.text(0.5, 0.5, "No image summaries to plot", ha="center", va="center")
		ax.set_axis_off()
		fig.tight_layout()
		fig.savefig(out_png, dpi=160, bbox_inches="tight")
		plt.close(fig)
		return

	image_names = [r["image"] for r in summary_rows]
	n_images = len(image_names)
	fig, axes = plt.subplots(1, n_images, figsize=(max(8.0, 2.9 * n_images), 4.8), squeeze=False)
	axes = axes[0]

	for ax, row in zip(axes, summary_rows):
		kept = int(row["kept_roots"])
		filtered = int(row["filtered_roots"])
		total = kept + filtered
		is_tilted = is_tilted_image(row["image"])
		in_color = "tab:orange" if is_tilted else "tab:green"

		if total <= 0:
			ax.text(0.5, 0.5, "No roots", ha="center", va="center")
			ax.set_axis_off()
			continue

		ax.pie(
			[kept, filtered],
			labels=["In", "Out"],
			colors=[in_color, "0.7"],
			autopct="%1.1f%%",
			startangle=90,
			counterclock=False,
			wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
			textprops={"fontsize": 8},
		)
		ax.set_title(f"{row['image']}\n(n={total})", fontsize=9)
		ax.axis("equal")

	fig.suptitle(f"Filtered fraction by image ({method_label})", y=0.98)
	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
	fig.savefig(out_png, dpi=180, bbox_inches="tight")
	plt.close(fig)


def normalize_param_values(values):
	if isinstance(values, (list, tuple, np.ndarray)):
		return [float(v) for v in values]
	return [float(values)]


def value_token(prefix, value):
	if float(value).is_integer():
		text = str(int(value))
	else:
		text = f"{value:.6g}"
	text = text.replace("-", "m").replace(".", "p")
	return f"{prefix}_{text}"


def summarize_and_filter(rows, peak_threshold):
	grouped = {}
	for row in rows:
		grouped.setdefault(row["image"], []).append(row)

	kept_rows = []
	summary_rows = []
	for image_name in sorted(grouped):
		img_rows = grouped[image_name]
		eligible_count = len(img_rows)
		filtered_nopeak = 0
		filtered_threshold = 0
		img_kept = []

		for row in img_rows:
			wl = float(row["wavelength_mm"])
			peak_height = float(row["peak_height"])
			if not np.isfinite(wl):
				filtered_nopeak += 1
				continue
			if (not np.isfinite(peak_height)) or (peak_height < peak_threshold):
				filtered_threshold += 1
				continue

			img_kept.append(
				{
					"image": row["image"],
					"root_label": row["root_label"],
					"wavelength_mm": wl,
					"wavelength_err_mm": float(row["wavelength_err_mm"]),
					"peak_lag_index": int(row["peak_lag_index"]),
					"peak_height": peak_height,
					"n_theta_samples": int(row["n_theta_samples"]),
					"main_length_mm": float(row["main_length_mm"]),
				}
			)

		filtered_count = filtered_nopeak + filtered_threshold
		frac = float(filtered_count / eligible_count) if eligible_count > 0 else np.nan
		summary_rows.append(
			{
				"image": image_name,
				"eligible_roots": int(eligible_count),
				"kept_roots": int(len(img_kept)),
				"filtered_roots": int(filtered_count),
				"filtered_no_peak": int(filtered_nopeak),
				"filtered_peak_threshold": int(filtered_threshold),
				"filtered_fraction": frac,
			}
		)
		kept_rows.extend(img_kept)

	return kept_rows, summary_rows


def run_for_threshold(autocorr_rows, base_output_dir, peak_threshold):
	tag = value_token("ct", peak_threshold)
	kept_rows, summary_rows = summarize_and_filter(autocorr_rows, peak_threshold=peak_threshold)

	errorbar_path = base_output_dir / "errorbars" / f"wavelength_errorbar_by_image_{tag}.png"
	frac_path = base_output_dir / "filtered_fraction" / f"filtered_fraction_by_image_{tag}.png"
	hist_path = base_output_dir / "wavelength_distributions" / f"wavelength_distribution_{tag}.png"
	plot_errorbars_by_image(kept_rows, errorbar_path)
	plot_filtered_fraction(kept_rows, summary_rows, frac_path, method_label=f"peak threshold >= {peak_threshold:.6g}")
	plot_histogram_combined(kept_rows, hist_path)

	kept_total = int(np.sum([r["kept_roots"] for r in summary_rows]))
	eligible_total = int(np.sum([r["eligible_roots"] for r in summary_rows]))
	filtered_total = int(np.sum([r["filtered_roots"] for r in summary_rows]))
	print(f"  eligible={eligible_total}, kept={kept_total}, filtered={filtered_total}")
	print(f"  saved plots with tag={tag} in {base_output_dir}")


def main():
	# Parameters (edit here, then run script directly)
	input_dir = Path("data/col_segmented/edited")
	base_output_dir = Path("src/dataWavelengthColParam/plots/cutoff_threshold")
	cache_path = Path("src/dataWavelengthColParam/outputs/autocorr_cache/autocorr_data.npz")

	dx_pixels = 8
	min_root_length = 200.0
	max_lag = 200
	force_rebuild_cache = False

	# Can be a scalar or a list/tuple/array.
	peak_threshold = [0.05, 0.1, 0.2, 0.25]

	autocorr_rows = ensure_autocorr_cache(
		cache_path=cache_path,
		input_dir=input_dir,
		dx_pixels=dx_pixels,
		min_root_length=min_root_length,
		max_lag=max_lag,
		force_rebuild=force_rebuild_cache,
	)
	if not autocorr_rows:
		print("No eligible roots found in cache data")
		return

	values = normalize_param_values(peak_threshold)
	print(f"Running peak-threshold analysis for {len(values)} value(s)")
	for val in values:
		print(f"\nPeak threshold = {val:.6g}")
		run_for_threshold(autocorr_rows, base_output_dir, peak_threshold=val)

	print("\nDone")


if __name__ == "__main__":
	main()
