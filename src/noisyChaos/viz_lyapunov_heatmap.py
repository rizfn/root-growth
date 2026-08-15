import argparse
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


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
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


def save_figure(fig, out_path: str) -> None:
    base, _ = os.path.splitext(out_path)
    fig.savefig(f"{base}.pdf", format="pdf")


def load_run_summaries(base_dir: str) -> pd.DataFrame:
    pattern = os.path.join(base_dir, "outputs", "lyapunov_scan", "run_summaries", "*.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No run summary files found at {pattern}")

    rows = []
    for fp in files:
        df = pd.read_csv(fp, sep="\t")
        if df.empty:
            continue
        rows.append(df.iloc[0])

    if not rows:
        raise RuntimeError("Run summary files exist but no rows could be loaded")

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["k", "eta"], as_index=False)
        .agg(
            lambda_mean=("lambda", "mean"),
            lambda_std=("lambda", "std"),
            n_runs=("lambda", "count"),
        )
    )
    grouped["lambda_sem"] = grouped["lambda_std"] / np.sqrt(grouped["n_runs"].clip(lower=1))
    return grouped


def build_grid(df: pd.DataFrame, value_col: str):
    ks = np.array(sorted(df["k"].unique()))
    etas = np.array(sorted(df["eta"].unique()))

    grid = np.full((len(etas), len(ks)), np.nan)
    for i, eta in enumerate(etas):
        sub = df[df["eta"] == eta].set_index("k")
        for j, k in enumerate(ks):
            if k in sub.index:
                grid[i, j] = sub.loc[k, value_col]

    return ks, etas, grid


def log_edges(vals: np.ndarray) -> np.ndarray:
    lv = np.log(vals)
    d = np.diff(lv)
    if len(d) == 0:
        return np.array([vals[0] / 1.1, vals[0] * 1.1])
    return np.exp(np.r_[lv[0] - d[0] / 2.0, lv[:-1] + d / 2.0, lv[-1] + d[-1] / 2.0])


def linear_edges(vals: np.ndarray) -> np.ndarray:
    d = np.diff(vals)
    if len(d) == 0:
        return np.array([vals[0] - 0.1, vals[0] + 0.1])
    return np.r_[vals[0] - d[0] / 2.0, vals[:-1] + d / 2.0, vals[-1] + d[-1] / 2.0]


# White (lambda = 0, periodic) to red (lambda > 0, chaotic), linear.
# Taken as the upper half of RdBu_r so the saturation ramp matches the old diverging map.
WHITE_RED = LinearSegmentedColormap.from_list(
    "white_red", plt.get_cmap("RdBu_r")(np.linspace(0.5, 1.0, 256))
)

# Above the ~1e-4 sampling floor, well below the chaotic values ~0.5.
LAMBDA_THRESHOLD = 5e-3


def _build_sequential_norm(grid: np.ndarray):
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return None
    vmax = float(np.max(finite))
    if vmax <= 0.0:
        return None
    return Normalize(vmin=0.0, vmax=vmax)


def plot_lyapunov_value_heatmap(agg_df: pd.DataFrame, out_path: str):
    ks, etas, lambda_grid = build_grid(agg_df, "lambda_mean")
    k_edges = linear_edges(ks)
    eta_edges = log_edges(etas)

    cmap = WHITE_RED.copy()
    cmap.set_bad(color="lightgray")
    norm = _build_sequential_norm(lambda_grid)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    m = ax.pcolormesh(
        k_edges,
        eta_edges,
        np.ma.masked_invalid(lambda_grid),
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,  # no hairline seams between cells in the PDF
    )
    ax.set_xlabel(r"$k\,\tau$")
    ax.set_ylabel(r"$\eta$")
    ax.set_yscale("log")
    style_axes(ax)
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"$\lambda$")
    cb.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])

    save_figure(fig, out_path)
    plt.close(fig)


def plot_lyapunov_contour_heatmap(agg_df: pd.DataFrame, out_path: str, threshold: float = LAMBDA_THRESHOLD):
    """Filled-contour version, with the lambda = threshold line marking the chaotic boundary."""
    ks, etas, lambda_grid = build_grid(agg_df, "lambda_mean")
    grid = np.ma.masked_invalid(lambda_grid)

    cmap = WHITE_RED.copy()
    cmap.set_bad(color="lightgray")
    norm = _build_sequential_norm(lambda_grid)
    step = 0.05
    levels = np.arange(0.0, np.ceil(norm.vmax / step) * step + 0.5 * step, step)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    m = ax.contourf(ks, etas, grid, levels=levels, cmap=cmap, extend="min")
    m.set_edgecolor("face")  # hide seams between filled levels in vector output
    ax.contour(ks, etas, grid, levels=[threshold], colors="black", linewidths=2.5)
    ax.set_xlabel(r"$k\,\tau$")
    ax.set_ylabel(r"$\eta$")
    ax.set_yscale("log")
    style_axes(ax)
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"$\lambda$")
    cb.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])

    save_figure(fig, out_path)
    plt.close(fig)


def plot_lyapunov_sem_heatmap(agg_df: pd.DataFrame, out_path: str):
    ks, etas, sem_grid = build_grid(agg_df, "lambda_sem")
    k_edges = linear_edges(ks)
    eta_edges = log_edges(etas)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    m = ax.pcolormesh(
        k_edges,
        eta_edges,
        np.ma.masked_invalid(sem_grid),
        cmap="viridis",
        shading="auto",
    )
    ax.set_xlabel(r"$k\,\tau$")
    ax.set_ylabel(r"$\eta$")
    ax.set_yscale("log")
    style_axes(ax)
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"SEM$(\lambda)$")
    cb.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])

    save_figure(fig, out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot Lyapunov heatmap over (k, eta).")
    parser.add_argument("--base-dir", default=os.path.dirname(__file__), help="Directory containing outputs/lyapunov_scan")
    parser.add_argument(
        "--summary-out",
        default=os.path.join(os.path.dirname(__file__), "outputs", "lyapunov_scan", "cell_summary.tsv"),
        help="Output TSV for aggregated per-cell statistics",
    )
    parser.add_argument(
        "--plots-dir",
        default=os.path.join(os.path.dirname(__file__), "plots", "lyapunov"),
        help="Output directory for Lyapunov figures",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    raw_df = load_run_summaries(args.base_dir)
    agg_df = aggregate(raw_df)
    agg_df.sort_values(["eta", "k"], inplace=True)
    agg_df.to_csv(args.summary_out, sep="\t", index=False)

    set_plot_style()

    values_out = os.path.join(args.plots_dir, "lyapunov_values_heatmap.pdf")
    contour_out = os.path.join(args.plots_dir, f"lyapunov_values_heatmap_contour_lc{LAMBDA_THRESHOLD:g}.pdf")
    sem_out = os.path.join(args.plots_dir, "lyapunov_sem_heatmap.pdf")
    plot_lyapunov_value_heatmap(agg_df, values_out)
    plot_lyapunov_contour_heatmap(agg_df, contour_out)
    plot_lyapunov_sem_heatmap(agg_df, sem_out)
    print(f"Wrote summary: {args.summary_out}")
    print(f"Wrote figures: {values_out}, {contour_out}, {sem_out}")


if __name__ == "__main__":
    main()
