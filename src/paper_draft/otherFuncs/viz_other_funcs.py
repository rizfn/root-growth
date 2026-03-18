import glob
import math
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "otherResponseFunctions"))
OUT_DIR = os.path.join(SCRIPT_DIR, "plots")

TAU = 1.0
RECORD_DT = 0.1
TS_K_TARGET = 3.0
TS_WINDOW_TAU = 8.0
PERIOD_EST_WINDOW_TAU = 80.0
HOPF_LINE_COLOR = "#666666"

FUNCTIONS = {
	"lnp1": {
		"label": r"$\mathrm{sign}(x)\,\ln(1+|x|)$",
		"color": "#901A1E",
		"func": lambda x: np.sign(x) * np.log1p(np.abs(x)),
	},
	"tanh": {
		"label": r"$\tanh(x)$",
		"color": "#547AA5",
		"func": np.tanh,
	},
	"xexpnegx": {
		"label": r"$\mathrm{sign}(x)\,|x|e^{-|x|}$",
		"color": "#CBA810",
		"func": lambda x: np.sign(x) * np.abs(x) * np.exp(-np.abs(x)),
	},
}

# Data-coordinate positions for function labels on the first panel.
# Tweak these values to move each equation label.
FUNC_LABEL_POSITIONS = {
	"lnp1": (-2, 1.85),
	"tanh": (2.5, 0.7),
	"xexpnegx": (0.1, -0.18),
}

FUNC_EQUATION_LABELS = {
	"lnp1": r"$\mathrm{sgn}(x)\,\ln(1+|x|)$",
	"tanh": r"$\tanh(x)$",
	"xexpnegx": r"$\mathrm{sgn}(x)\,|x|e^{-|x|}$",
}


def load_timeseries(path):
	try:
		data = np.loadtxt(path, skiprows=1, usecols=(0, 1))
		if data.ndim == 1:
			data = data.reshape(1, 2)
		finite_mask = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
		if not np.any(finite_mask):
			return None, None
		return data[finite_mask, 0], data[finite_mask, 1]
	except Exception:
		return None, None


def extract_k(path):
	match = re.search(r"/k_([^_]+)_ic_", path)
	if not match:
		return None
	try:
		return float(match.group(1))
	except ValueError:
		return None


def load_func_sweep(func_name):
	pattern = os.path.join(SOURCE_DIR, "outputs", func_name, "k_*.tsv")
	files = sorted(glob.glob(pattern))
	if not files:
		return np.array([]), np.array([]), []

	ks = []
	amps = []
	paths = []
	for path in files:
		k = extract_k(path)
		if k is None:
			continue
		_, theta = load_timeseries(path)
		if theta is None or len(theta) == 0:
			amp = np.nan
		else:
			amp = np.max(np.abs(theta[len(theta) // 2 :]))
		ks.append(k)
		amps.append(amp)
		paths.append(path)

	if not ks:
		return np.array([]), np.array([]), []

	ks = np.array(ks)
	amps = np.array(amps)
	idx = np.argsort(ks)
	return ks[idx], amps[idx], [paths[i] for i in idx]


def estimate_period(t, y):
	if t is None or y is None or len(t) < 5 or len(y) < 5:
		return np.nan

	n_tail = int(round(PERIOD_EST_WINDOW_TAU * TAU / RECORD_DT))
	n_tail = max(10, min(len(t), n_tail))
	t_tail = t[-n_tail:]
	y_tail = y[-n_tail:]

	y_centered = y_tail - np.mean(y_tail)
	if np.allclose(y_centered, 0.0):
		return np.nan

	crossings = []
	for i in range(len(y_centered) - 1):
		y0 = y_centered[i]
		y1 = y_centered[i + 1]
		if y0 <= 0.0 < y1:
			dy = y1 - y0
			if dy == 0.0:
				continue
			frac = -y0 / dy
			crossings.append(t_tail[i] + frac * (t_tail[i + 1] - t_tail[i]))

	if len(crossings) < 3:
		return np.nan

	periods = np.diff(crossings)
	finite_periods = periods[np.isfinite(periods)]
	if len(finite_periods) == 0:
		return np.nan
	return float(np.median(finite_periods))


def last_zero_crossing_before(t, y, t_ref):
	if t is None or y is None or len(t) < 2 or len(y) < 2:
		return None
	last_cross = None
	for i in range(len(t) - 1):
		t0, t1 = t[i], t[i + 1]
		y0, y1 = y[i], y[i + 1]
		if t0 > t_ref:
			break
		if y0 == 0.0 and t0 <= t_ref:
			last_cross = float(t0)
			continue
		if y1 == 0.0 and t1 <= t_ref:
			last_cross = float(t1)
			continue
		if (y0 < 0.0 and y1 > 0.0) or (y0 > 0.0 and y1 < 0.0):
			dy = y1 - y0
			if dy == 0.0:
				continue
			frac = -y0 / dy
			t_cross = t0 + frac * (t1 - t0)
			if t_cross <= t_ref:
				last_cross = float(t_cross)
	return last_cross


def nearest_index(values, target):
	if len(values) == 0:
		return None
	return int(np.argmin(np.abs(values - target)))


def format_tick_value(v):
	if np.isclose(v, round(v), rtol=0.0, atol=1e-10):
		return f"{int(round(v)):,}"
	return f"{v:.3g}"


def keep_only_selected_xticklabels(ax, targets, labels):
	# Keep all tick marks, ensure target x positions exist as ticks, and label only targets.
	xticks = ax.get_xticks()
	xlim = ax.get_xlim()
	xmin, xmax = min(xlim), max(xlim)
	if len(targets) != len(labels):
		return

	visible_xticks = xticks[(xticks >= xmin) & (xticks <= xmax)]
	targets_in_view = [t for t in targets if xmin <= t <= xmax]
	if len(visible_xticks) == 0 and len(targets_in_view) == 0:
		return

	all_xticks = np.array(sorted(set(np.concatenate((visible_xticks, np.array(targets_in_view))))))
	ax.xaxis.set_major_locator(FixedLocator(all_xticks))

	x_tol = 1e-9 * max(1.0, *(abs(t) for t in targets))

	def xfmt(val, _):
		for target, label in zip(targets, labels):
			if np.isclose(val, target, rtol=0.0, atol=x_tol):
				return label
		return ""

	ax.xaxis.set_major_formatter(FuncFormatter(xfmt))


def keep_only_endpoint_ticklabels(ax):
	# Keep all tick marks for visual guidance, but label only the first and last y ticks.

	yticks = ax.get_yticks()
	ylim = ax.get_ylim()
	ymin, ymax = min(ylim), max(ylim)
	visible_yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
	if len(visible_yticks) >= 2:
		y0 = visible_yticks[0]
		y1 = visible_yticks[-1]

		def yfmt(val, _):
			if np.isclose(val, y0, rtol=0.0, atol=1e-10):
				return format_tick_value(val)
			if np.isclose(val, y1, rtol=0.0, atol=1e-10):
				return format_tick_value(val)
			return ""

		ax.yaxis.set_major_formatter(FuncFormatter(yfmt))


def keep_only_selected_yticklabels(ax, targets, labels):
	yticks = ax.get_yticks()
	ylim = ax.get_ylim()
	ymin, ymax = min(ylim), max(ylim)
	if len(targets) != len(labels):
		return

	visible_yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
	targets_in_view = [t for t in targets if ymin <= t <= ymax]
	if len(visible_yticks) == 0 and len(targets_in_view) == 0:
		return

	all_yticks = np.array(sorted(set(np.concatenate((visible_yticks, np.array(targets_in_view))))))
	ax.yaxis.set_major_locator(FixedLocator(all_yticks))

	y_tol = 1e-9 * max(1.0, *(abs(t) for t in targets))

	def yfmt(val, _):
		for target, label in zip(targets, labels):
			if np.isclose(val, target, rtol=0.0, atol=y_tol):
				return label
		return ""

	ax.yaxis.set_major_formatter(FuncFormatter(yfmt))


def main():
	plt.rcParams.update(
		{
			"figure.dpi": 150,
			"savefig.dpi": 300,
			"font.size": 24,
			"axes.titlesize": 24,
			"axes.labelsize": 32,
			"xtick.labelsize": 24,
			"ytick.labelsize": 24,
			"legend.fontsize": 22,
			"axes.grid": True,
			"grid.alpha": 0.3,
			"grid.linewidth": 0.7,
			"axes.spines.top": True,
			"axes.spines.right": True,
		}
	)

	sweep_data = {}
	for func_name in FUNCTIONS:
		ks, amps, paths = load_func_sweep(func_name)
		if len(ks) == 0:
			print(
				f"No output files found for {func_name} in "
				f"{os.path.join(SOURCE_DIR, 'outputs', func_name)}"
			)
			return
		sweep_data[func_name] = (ks, amps, paths)

	fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
	fig.set_constrained_layout_pads(wspace=0.08, hspace=0.02)
	fig.patch.set_alpha(0.0)

	ax_func, ax_ts, ax_amp = axes

	x = np.linspace(-6.0, 6.0, 800)
	for func_name, meta in FUNCTIONS.items():
		y = meta["func"](x)
		ax_func.plot(x, y, lw=2.2, color=meta["color"], label=meta["label"])
		if func_name in FUNC_LABEL_POSITIONS and func_name in FUNC_EQUATION_LABELS:
			x_text, y_text = FUNC_LABEL_POSITIONS[func_name]
			ax_func.text(
				x_text,
				y_text,
				FUNC_EQUATION_LABELS[func_name],
				color=meta["color"],
				fontsize=22,
				ha="left",
				va="center",
			)
	ax_func.axhline(0.0, color="black", lw=0.7, alpha=0.55)
	ax_func.axvline(0.0, color="black", lw=0.7, alpha=0.55)
	ax_func.set_xlim(-6.0, 6.0)
	ax_func.set_xlabel(r"$x$", labelpad=-16)
	ax_func.set_ylabel(r"$f(x)$", labelpad=-22)
	# ax_func.legend(frameon=False, loc="upper left")

	n_window = max(10, int(round(TS_WINDOW_TAU * TAU / RECORD_DT)))
	window_span = TS_WINDOW_TAU * TAU
	used_k_values = []
	for func_name, meta in FUNCTIONS.items():
		ks, _, paths = sweep_data[func_name]
		idx = nearest_index(ks, TS_K_TARGET)
		if idx is None:
			continue

		t, theta = load_timeseries(paths[idx])
		if t is None or theta is None or len(t) == 0:
			continue

		period = estimate_period(t, theta)
		t_end = t[-1]
		t_window_start = t_end - window_span
		t_zero = last_zero_crossing_before(t, theta, t_window_start)

		if t_zero is None:
			t_seg = t[-n_window:]
			y_seg = theta[-n_window:]
			t_rel = t_seg - t_seg[0]
		else:
			mask = (t >= t_zero) & (t <= t_zero + window_span)
			t_seg = t[mask]
			y_seg = theta[mask]

			if len(t_seg) < 2:
				t_seg = t[-n_window:]
				y_seg = theta[-n_window:]
				t_rel = t_seg - t_seg[0]
			else:
				# Start all traces at the same phase anchor (zero crossing).
				t_seg = np.insert(t_seg, 0, t_zero)
				y_seg = np.insert(y_seg, 0, 0.0)
				t_rel = t_seg - t_zero
		used_k_values.append(ks[idx])

		if np.isfinite(period):
			legend_label = rf"{meta['label']}, $T\approx{period:.2f}$"
		else:
			legend_label = rf"{meta['label']}, $T\approx\mathrm{{n/a}}$"

		ax_ts.plot(t_rel, y_seg, lw=1.7, color=meta["color"], label=legend_label)

	if used_k_values:
		used_k = float(np.median(used_k_values))
	else:
		used_k = TS_K_TARGET
	ax_ts.set_xlim(0.0, 8.0)
	ax_ts.set_xlabel(r"$t$", labelpad=-16)
	ax_ts.set_ylabel(r"$\theta(t)$", labelpad=-20)
	# ax_ts.legend(frameon=False, loc="best")

	for func_name, meta in FUNCTIONS.items():
		ks, amps, _ = sweep_data[func_name]
		ax_amp.plot(ks, amps, "o-", ms=3, lw=1.6, color=meta["color"], label=meta["label"])

	ax_amp.axvline(
		math.pi / 2,
		ls=(0, (8, 6)),
		lw=1.0,
		color=HOPF_LINE_COLOR,
		alpha=0.75,
		label=rf"$k_{{\rm Hopf}}={math.pi/2:.3f}$",
	)
	ymin_amp, ymax_amp = ax_amp.get_ylim()
	y_text = ymin_amp + 0.88 * (ymax_amp - ymin_amp)
	ax_amp.text(
		math.pi / 2 + 0.08,
		y_text,
		r"$k\tau=\pi/2$",
		color=HOPF_LINE_COLOR,
		fontsize=20,
		ha="left",
		va="center",
	)
	ax_amp.set_xlabel(r"$k\tau$", labelpad=-16)
	ax_amp.set_ylabel(r"$\max|\theta|$", labelpad=-16)
	# ax_amp.legend(frameon=False, loc="upper left")

	for ax in axes:
		ax.set_facecolor("none")
		for spine in ax.spines.values():
			spine.set_visible(True)
			spine.set_color("black")
			spine.set_linewidth(0.9)
		if ax is ax_func:
			# Make the first panel grid coarser so cells are visibly larger.
			ax.xaxis.set_major_locator(MultipleLocator(8.0))
			ax.yaxis.set_major_locator(MultipleLocator(2.0))
		if ax is ax_ts:
			ax.xaxis.set_major_locator(MultipleLocator(8.0))
			ax.yaxis.set_major_locator(MultipleLocator(8.0))
		if ax is ax_amp:
			ax.xaxis.set_major_locator(MultipleLocator(8.0))
			ax.yaxis.set_major_locator(MultipleLocator(8.0))
		ax.minorticks_on()
		ax.grid(True, which="both", alpha=0.3)
		if ax is ax_ts:
			keep_only_selected_xticklabels(ax, targets=(0.0, 8.0), labels=("0", "8"))
			keep_only_selected_yticklabels(ax, targets=(-3, 3), labels=("-3", "3"))
		elif ax is ax_amp:
			# Pull endpoints a bit closer to the frame while keeping autoscaling behavior.
			ax.margins(x=0.01)
			keep_only_selected_xticklabels(ax, targets=(1.0, 5.0), labels=("1", "5"))
			keep_only_endpoint_ticklabels(ax)
		else:
			keep_only_selected_xticklabels(ax, targets=(-6.0, 6.0), labels=("-6", "6"))
			keep_only_endpoint_ticklabels(ax)
		ax.tick_params(
			axis="both",
			which="both",
			direction="out",
			top=False,
			right=False,
			pad=1,
		)

	os.makedirs(OUT_DIR, exist_ok=True)
	out_path = os.path.join(OUT_DIR, "other_functions_comparison.svg")
	fig.savefig(out_path, format="svg", transparent=True, facecolor="none", edgecolor="none")
	print(f"Saved: {out_path}")
	fig.savefig(out_path.replace(".svg", ".pdf"), format="pdf", transparent=True, facecolor="none", edgecolor="none")


if __name__ == "__main__":
	main()
