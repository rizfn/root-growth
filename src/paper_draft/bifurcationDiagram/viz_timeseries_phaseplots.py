"""Plot 4 parameter sets, each with phase portrait (left) and timeseries (right).

Expected input files are produced by runTimeseriesPhaseplots.sh and have names:
  tauk_<real>_target_<target>_ic_<tag>.tsv
"""

import glob
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "outputs" / "timeseries_phaseplots"
PLOT_DIR = SCRIPT_DIR / "plots" / "timeseries_phaseplots"

TAU = 1.0

IC_PALETTE = {
	1.0: "#2166AC",
	-1.0: "#92C5DE",
	2.0: "#D6604D",
	-2.0: "#F4A582",
}

# Panel order is read left-to-right, top-to-bottom for selective IC rules.
# Left side: LC2-focused parameters (3.60, 4.13). Right side: high-k (4.80, 4.90).
PANEL_SPECS = [
	{"tauk": 3.60, "row": 0, "phase_col": 0, "ts_col": 1, "ics": [1.0]},
	{"tauk": 4.13, "row": 1, "phase_col": 0, "ts_col": 1, "ics": [1.0, -1.0, 2.0, -2.0]},
	{"tauk": 4.80, "row": 0, "phase_col": 2, "ts_col": 3, "ics": [1.0, -1.0]},
	{"tauk": 4.90, "row": 1, "phase_col": 2, "ts_col": 3, "ics": [1.0, -1.0]},
]


def align_to_upward_zero(theta: np.ndarray, t: np.ndarray, window: float) -> Tuple[np.ndarray, np.ndarray]:
	if theta is None or t is None or len(theta) < 3 or len(t) < 3:
		return np.array([]), np.array([])

	t_end = float(t[-1])
	t_ref = t_end - window

	last_cross = None
	for i in range(len(theta) - 1):
		y0 = float(theta[i])
		y1 = float(theta[i + 1])
		if not (y0 < 0.0 <= y1):
			continue
		t0 = float(t[i])
		t1 = float(t[i + 1])
		dy = y1 - y0
		if abs(dy) < 1e-15:
			t_cross = t0
		else:
			frac = -y0 / dy
			t_cross = t0 + frac * (t1 - t0)
		if t_cross <= t_ref:
			last_cross = t_cross

	if last_cross is None:
		# Fallback: no suitable phase anchor found, use the plain last window.
		t_start = max(float(t[0]), t_ref)
	else:
		# User-requested rule: start at the crossing before (end - window).
		t_start = last_cross

	t_stop = min(t_end, t_start + window)
	mask = (t >= t_start) & (t <= t_stop)
	if np.sum(mask) < 3:
		return np.array([]), np.array([])

	theta_sel = theta[mask]
	t_rel = t[mask] - t_start
	return theta_sel, t_rel


def parse_real_tauk(path: Path) -> float | None:
	m = re.search(r"tauk_([0-9n]+\.[0-9]+)", path.name)
	if not m:
		return None
	token = m.group(1).replace("n", "-")
	try:
		return float(token)
	except ValueError:
		return None


def parse_ic(path: Path) -> float | None:
	m = re.search(r"_ic_([0-9n]+\.[0-9]+)\.tsv$", path.name)
	if not m:
		return None
	token = m.group(1).replace("n", "-")
	try:
		return float(token)
	except ValueError:
		return None


def load_ts(path: Path) -> Tuple[np.ndarray | None, np.ndarray | None]:
	try:
		arr = np.loadtxt(path, skiprows=1)
	except Exception:
		return None, None
	if arr.ndim != 2 or arr.shape[0] < 3:
		return None, None
	return arr[:, 0], arr[:, 1]


def collect_files() -> Dict[Tuple[float, float], Path]:
	mapping: Dict[Tuple[float, float], Path] = {}
	for fp in glob.glob(str(DATA_DIR / "*.tsv")):
		p = Path(fp)
		tauk = parse_real_tauk(p)
		ic = parse_ic(p)
		if tauk is None or ic is None:
			continue
		mapping[(round(tauk, 3), ic)] = p
	return mapping


def set_style() -> None:
	plt.rcParams.update(
		{
			"figure.dpi": 150,
			"savefig.dpi": 300,
			"font.size": 18,
			"axes.titlesize": 18,
			"axes.labelsize": 20,
			"xtick.labelsize": 14,
			"ytick.labelsize": 14,
			"legend.fontsize": 11,
			"axes.grid": True,
			"grid.alpha": 0.3,
			"grid.linewidth": 0.7,
			"axes.spines.top": True,
			"axes.spines.right": True,
		}
	)


def style_axes(ax: plt.Axes) -> None:
	ax.set_facecolor("none")
	for spine in ax.spines.values():
		spine.set_visible(True)
		spine.set_color("black")
		spine.set_linewidth(0.9)
	ax.minorticks_on()
	ax.grid(True, which="major", alpha=0.3, linewidth=0.7)
	ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=1)


def main() -> None:
	set_style()

	if not DATA_DIR.exists():
		raise RuntimeError(f"Missing data directory: {DATA_DIR}")

	files = collect_files()
	if not files:
		raise RuntimeError(f"No timeseries TSV files found in {DATA_DIR}")

	fig, axes = plt.subplots(
		2,
		4,
		figsize=(24, 10),
		constrained_layout=True,
		gridspec_kw={"wspace": 0.12, "hspace": 0.12},
	)
	fig.set_constrained_layout_pads(wspace=0.04, hspace=0.02)
	fig.patch.set_alpha(0.0)

	window = 4.0 * 4.0 * TAU

	for spec in PANEL_SPECS:
		tauk = spec["tauk"]
		k_real = tauk / TAU
		tauk_disp = round(tauk, 3)
		k_disp = round(k_real, 3)
		ics = spec["ics"]
		ax_phase = axes[spec["row"], spec["phase_col"]]
		ax_ts = axes[spec["row"], spec["ts_col"]]

		for ic in ics:
			path = files.get((round(tauk, 3), ic))
			if path is None:
				continue

			t, theta = load_ts(path)
			if t is None or theta is None:
				continue

			th_aligned, x_ts = align_to_upward_zero(theta, t, window)
			if th_aligned.size < 3 or x_ts.size < 3:
				continue
			color = IC_PALETTE.get(ic, "gray")

			dtheta = np.gradient(th_aligned, x_ts)
			ax_phase.plot(
				th_aligned,
				dtheta,
				lw=1.0,
				color=color,
				alpha=0.9,
				label=f"IC = {ic:+.1f}",
			)

			ax_ts.plot(
				x_ts,
				th_aligned,
				lw=1.1,
				color=color,
				alpha=0.9,
				label=f"IC = {ic:+.1f}",
			)

		ax_phase.set_title(rf"$\tau k={tauk_disp:.3f},\ k={k_disp:.3f}$")
		ax_phase.set_xlabel(r"$\theta$")
		ax_phase.set_ylabel(r"$d\theta/dt$")
		style_axes(ax_phase)

		ax_ts.set_xlabel(r"$t$")
		ax_ts.set_ylabel(r"$\theta(t)$")
		style_axes(ax_ts)

	PLOT_DIR.mkdir(parents=True, exist_ok=True)
	out_svg = PLOT_DIR / "timeseries_phaseplots_4params.svg"
	out_pdf = PLOT_DIR / "timeseries_phaseplots_4params.pdf"
	fig.savefig(out_svg, format="svg", transparent=True, facecolor="none", edgecolor="none")
	fig.savefig(out_pdf, format="pdf", transparent=True, facecolor="none", edgecolor="none")
	plt.close(fig)
	print(f"Saved: {out_svg}")
	print(f"Saved: {out_pdf}")


if __name__ == "__main__":
	main()
