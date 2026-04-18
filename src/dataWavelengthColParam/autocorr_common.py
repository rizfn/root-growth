from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
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


def first_peak_uncertainty_mm(ac, peak_lag, dx_mm):
	"""Estimate wavelength uncertainty from first-peak width."""
	if ac is None or peak_lag is None or peak_lag <= 0 or peak_lag >= len(ac):
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


def _cache_meta(input_dir, dx_pixels, min_root_length, max_lag):
	return {
		"input_dir": str(input_dir),
		"dx_pixels": int(dx_pixels),
		"min_root_length": float(min_root_length),
		"max_lag": int(max_lag),
	}


def save_autocorr_cache(rows, cache_path, meta):
	"""Save computed per-root autocorrelation rows to a compressed npz cache."""
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		np.savez_compressed(cache_path, meta_json=np.array(json.dumps(meta)), empty=np.array([1]))
		return

	images = np.array([r["image"] for r in rows], dtype=str)
	root_labels = np.array([r["root_label"] for r in rows], dtype=int)
	main_length_mm = np.array([r["main_length_mm"] for r in rows], dtype=float)
	n_theta_samples = np.array([r["n_theta_samples"] for r in rows], dtype=int)
	dx_mm = np.array([r["dx_mm"] for r in rows], dtype=float)
	peak_lag_index = np.array([r["peak_lag_index"] for r in rows], dtype=int)
	peak_height = np.array([r["peak_height"] for r in rows], dtype=float)
	wavelength_mm = np.array([r["wavelength_mm"] for r in rows], dtype=float)
	wavelength_err_mm = np.array([r["wavelength_err_mm"] for r in rows], dtype=float)
	ac_values = np.array([np.asarray(r["ac"], dtype=float) for r in rows], dtype=object)

	np.savez_compressed(
		cache_path,
		meta_json=np.array(json.dumps(meta)),
		images=images,
		root_labels=root_labels,
		main_length_mm=main_length_mm,
		n_theta_samples=n_theta_samples,
		dx_mm=dx_mm,
		peak_lag_index=peak_lag_index,
		peak_height=peak_height,
		wavelength_mm=wavelength_mm,
		wavelength_err_mm=wavelength_err_mm,
		ac_values=ac_values,
	)


def load_autocorr_cache(cache_path):
	"""Load cached per-root autocorrelation rows and metadata."""
	with np.load(cache_path, allow_pickle=True) as data:
		meta = json.loads(str(data["meta_json"].item()))
		if "images" not in data:
			return [], meta

		rows = []
		for i in range(len(data["images"])):
			rows.append(
				{
					"image": str(data["images"][i]),
					"root_label": int(data["root_labels"][i]),
					"main_length_mm": float(data["main_length_mm"][i]),
					"n_theta_samples": int(data["n_theta_samples"][i]),
					"dx_mm": float(data["dx_mm"][i]),
					"peak_lag_index": int(data["peak_lag_index"][i]),
					"peak_height": float(data["peak_height"][i]),
					"wavelength_mm": float(data["wavelength_mm"][i]),
					"wavelength_err_mm": float(data["wavelength_err_mm"][i]),
					"ac": np.asarray(data["ac_values"][i], dtype=float),
				}
			)
	return rows, meta


def compute_autocorr_rows(input_dir, dx_pixels, min_root_length, max_lag):
	"""Compute eligible-root autocorrelation rows for all images in input_dir."""
	files = sorted(input_dir.glob("*.png"))
	if not files:
		print(f"No PNG files found in {input_dir}")
		return []

	all_rows = []
	print(f"Computing autocorrelation cache from {len(files)} images")
	for filepath in files:
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
		count_for_image = 0

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

			x = np.unwrap(np.asarray(thetas, dtype=float))
			ac = compute_autocorr(x, max_lag=max_lag)
			peak_lag = first_positive_peak_lag(ac)

			if peak_lag is None:
				peak_lag_index = -1
				peak_height = np.nan
				wavelength_mm = np.nan
				wavelength_err_mm = np.nan
			else:
				peak_lag_index = int(peak_lag)
				peak_height = float(ac[peak_lag]) if peak_lag < len(ac) else np.nan
				wavelength_mm = float(peak_lag * dx_mm)
				wavelength_err_mm = float(first_peak_uncertainty_mm(ac, peak_lag, dx_mm))

			all_rows.append(
				{
					"image": filepath.name,
					"root_label": int(region.label),
					"main_length_mm": float(main_length * mm_per_pixel),
					"n_theta_samples": int(len(thetas)),
					"dx_mm": float(dx_mm),
					"peak_lag_index": peak_lag_index,
					"peak_height": peak_height,
					"wavelength_mm": wavelength_mm,
					"wavelength_err_mm": wavelength_err_mm,
					"ac": np.asarray(ac, dtype=float),
				}
			)
			count_for_image += 1

		print(f"  eligible roots cached: {count_for_image}")

	print(f"Total eligible roots cached: {len(all_rows)}")
	return all_rows


def ensure_autocorr_cache(cache_path, input_dir, dx_pixels, min_root_length, max_lag, force_rebuild=False):
	"""Load cache when valid; otherwise recompute and save a new cache."""
	expected = _cache_meta(input_dir, dx_pixels, min_root_length, max_lag)

	if (not force_rebuild) and cache_path.exists():
		try:
			rows, meta = load_autocorr_cache(cache_path)
			if meta == expected:
				print(f"Loaded autocorrelation cache: {cache_path}")
				return rows
			print("Autocorrelation cache metadata changed; rebuilding cache")
		except Exception:
			print("Failed to read autocorrelation cache; rebuilding cache")

	rows = compute_autocorr_rows(input_dir, dx_pixels, min_root_length, max_lag)
	save_autocorr_cache(rows, cache_path, expected)
	print(f"Saved autocorrelation cache: {cache_path}")
	return rows