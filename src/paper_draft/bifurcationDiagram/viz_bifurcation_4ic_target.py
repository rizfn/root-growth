import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "outputs", "bifurcation4IC_target")
PLOT_ROOT = os.path.join(SCRIPT_DIR, "plots")
PLOT_DIR = os.path.join(PLOT_ROOT, "total")

BRANCH_META: Dict[str, Dict[str, str]] = {
    "p4_pos": {"label": r"$\theta_0=+1$", "color": "#901A1E"},
    "p4_neg": {"label": r"$\theta_0=-1$", "color": "#901A1E"},
    "lc2_pos": {"label": r"$\theta_0=+2$", "color": "#901A1E"},
    "lc2_neg": {"label": r"$\theta_0=-2$", "color": "#901A1E"},
}

HOPF_COLOR = "#666666"
TWO_PI = 2.0 * np.pi


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 17,
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


def parse_branch(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"(p4_pos|p4_neg|lc2_pos|lc2_neg)_", base)
    return m.group(1) if m else "unknown"


def load_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # Skip header-only files (common when no maxima are detected).
    try:
        with open(path, "r", encoding="utf-8") as f:
            _header = f.readline()
            first_data = f.readline()
            if first_data.strip() == "":
                return np.array([]), np.array([])
    except OSError:
        return np.array([]), np.array([])

    try:
        arr = np.loadtxt(path, delimiter="\t", skiprows=1)
    except Exception:
        return np.array([]), np.array([])

    if arr.ndim == 1 and arr.size == 5:
        arr = arr.reshape(1, 5)
    if arr.size == 0:
        return np.array([]), np.array([])

    tauk_real = arr[:, 0]
    # Periodic wrapping: values like 2*pi + delta are shown as delta.
    theta_max = np.mod(arr[:, 4], TWO_PI)
    finite = np.isfinite(tauk_real) & np.isfinite(theta_max)
    return tauk_real[finite], theta_max[finite]


def main() -> None:
    set_plot_style()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tsv")))
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}")

    grouped: Dict[str, List[str]] = {k: [] for k in BRANCH_META}
    for fp in files:
        branch = parse_branch(fp)
        if branch in grouped:
            grouped[branch].append(fp)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    for branch, paths in grouped.items():
        if not paths:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for p in paths:
            x, y = load_points(p)
            if x.size == 0:
                continue
            xs.extend(x.tolist())
            ys.extend(y.tolist())
        if not xs:
            continue

        meta = BRANCH_META[branch]
        # Short vertical line markers are clearer than pixel squares in dense clouds.
        ax.plot(
            np.array(xs),
            np.array(ys),
            linestyle="None",
            marker=".",
            markersize=0.1,
            markeredgewidth=0.7,
            color=meta["color"],
            alpha=0.50,
            label=meta["label"],
        )

    hopf = np.pi / 2.0
    ax.axvline(hopf, color=HOPF_COLOR, lw=1.0, ls="--", alpha=0.75, label=r"$k\tau=\pi/2$")

    ax.set_xlabel(r"$k\tau$")
    ax.set_ylabel(r"$\theta_{\max}$")
    ax.set_ylim(0.0, TWO_PI)
    ax.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, TWO_PI])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    style_axes(ax)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out_png = os.path.join(PLOT_DIR, "bifurcation_4ic_target.png")
    fig.savefig(out_png, format="png", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
