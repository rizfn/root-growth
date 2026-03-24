import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(r"tau_([\deE+\-.]+)_k_([\deE+\-.]+)_eta_([\deE+\-.]+)_seed_(\d+)\.tsv$")


def parse_from_name(path: str):
    m = RUN_RE.search(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))


def load_laminar_lengths(base_dir: str) -> pd.DataFrame:
    pattern = os.path.join(base_dir, "outputs", "laminar_scan", "laminar_lengths", "*.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No laminar length files found at {pattern}")

    rows = []
    for fp in files:
        parsed = parse_from_name(fp)
        if parsed is None:
            continue
        tau, k, eta, seed = parsed
        df = pd.read_csv(fp, sep="\t")
        if "laminar_length" not in df.columns:
            continue
        valid = df["laminar_length"].dropna().values
        for v in valid:
            rows.append((tau, k, eta, seed, float(v)))

    if not rows:
        raise RuntimeError("No laminar length samples could be loaded")

    return pd.DataFrame(rows, columns=["tau", "k", "eta", "seed", "laminar_length"])


def ccdf(values: np.ndarray):
    x = np.sort(values)
    n = len(x)
    y = 1.0 - np.arange(1, n + 1) / float(n)
    return x, y


def pooled_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["k", "eta"], as_index=False)
        .agg(
            n_samples=("laminar_length", "count"),
            laminar_mean=("laminar_length", "mean"),
            laminar_median=("laminar_length", "median"),
            laminar_p90=("laminar_length", lambda x: np.quantile(x, 0.90)),
            laminar_p99=("laminar_length", lambda x: np.quantile(x, 0.99)),
        )
    )
    return out


def nearest(vals: np.ndarray, target: float) -> float:
    return float(vals[np.argmin(np.abs(vals - target))])


def _slice_sub(df: pd.DataFrame, col: str, value: float) -> pd.DataFrame:
    return df[np.isclose(df[col], value, rtol=1e-8, atol=1e-12)]


def count_valid_curves(df: pd.DataFrame, fixed_col: str, fixed_value: float, varying_col: str, min_points_per_curve: int) -> int:
    sub = _slice_sub(df, fixed_col, fixed_value)
    var_values = np.array(sorted(sub[varying_col].unique()))
    count = 0
    for v in var_values:
        vals = _slice_sub(sub, varying_col, v)["laminar_length"].values
        if len(vals) >= min_points_per_curve:
            count += 1
    return count


def choose_fixed_value(df: pd.DataFrame, fixed_col: str, preferred_value: float, varying_col: str, min_points_per_curve: int):
    fixed_values = np.array(sorted(df[fixed_col].unique()))
    preferred = nearest(fixed_values, preferred_value)
    preferred_count = count_valid_curves(df, fixed_col, preferred, varying_col, min_points_per_curve)
    if preferred_count >= 2:
        return preferred, preferred_count, False

    counts = []
    for v in fixed_values:
        counts.append(count_valid_curves(df, fixed_col, float(v), varying_col, min_points_per_curve))
    counts = np.array(counts)

    best_count = int(np.max(counts))
    best_candidates = fixed_values[counts == best_count]
    best = nearest(best_candidates, preferred)
    return best, best_count, (best != preferred)


def make_slice_plot(
    df: pd.DataFrame,
    fixed_col: str,
    fixed_value: float,
    varying_col: str,
    out_path: str,
    max_curves: int = 6,
    min_points_per_curve: int = 3,
):
    sub = _slice_sub(df, fixed_col, fixed_value)
    var_values = np.array(sorted(sub[varying_col].unique()))
    if len(var_values) == 0:
        return 0

    if len(var_values) > max_curves:
        idx = np.linspace(0, len(var_values) - 1, max_curves).astype(int)
        var_values = var_values[idx]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plotted = 0
    for v in var_values:
        vals = _slice_sub(sub, varying_col, v)["laminar_length"].values
        if len(vals) < min_points_per_curve:
            continue
        x, y = ccdf(vals)
        label = f"{varying_col}={v:.3g} (n={len(vals)})"
        ax.plot(x, y, linewidth=1.6, label=label)
        plotted += 1

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Laminar length")
    ax.set_ylabel("CCDF")
    ax.set_title(f"Laminar CCDF at {fixed_col}={fixed_value:.4g}")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No curves met min sample threshold", transform=ax.transAxes, ha="center", va="center")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return plotted


def heatmap_from_stats(stats: pd.DataFrame, value_col: str, title: str, out_path: str):
    ks = np.array(sorted(stats["k"].unique()))
    etas = np.array(sorted(stats["eta"].unique()))
    grid = np.full((len(etas), len(ks)), np.nan)

    for i, eta in enumerate(etas):
        sub = stats[np.isclose(stats["eta"], eta)].set_index("k")
        for j, k in enumerate(ks):
            if k in sub.index:
                grid[i, j] = sub.loc[k, value_col]

    def k_edges(vals):
        d = np.diff(vals)
        if len(d) == 0:
            return np.array([vals[0] - 0.1, vals[0] + 0.1])
        return np.r_[vals[0] - d[0] / 2, vals[:-1] + d / 2, vals[-1] + d[-1] / 2]

    def eta_edges(vals):
        lv = np.log(vals)
        d = np.diff(lv)
        if len(d) == 0:
            return np.array([vals[0] / 1.1, vals[0] * 1.1])
        return np.exp(np.r_[lv[0] - d[0] / 2, lv[:-1] + d / 2, lv[-1] + d[-1] / 2])

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    m = ax.pcolormesh(
        k_edges(ks),
        eta_edges(etas),
        np.ma.masked_invalid(grid),
        cmap="magma",
        shading="auto",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("eta")
    ax.set_yscale("log")
    ax.set_title(title)
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(value_col)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot laminar distributions and parameter dependence")
    parser.add_argument("--base-dir", default=os.path.dirname(__file__), help="Directory containing outputs/laminar_scan")
    parser.add_argument(
        "--stats-out",
        default=os.path.join(os.path.dirname(__file__), "outputs", "laminar_scan", "laminar_cell_stats.tsv"),
        help="Output TSV for pooled laminar stats",
    )
    parser.add_argument(
        "--plots-dir",
        default=os.path.join(os.path.dirname(__file__), "plots", "laminar"),
        help="Directory for generated figures",
    )
    parser.add_argument("--fixed-k", type=float, default=4.5, help="Preferred k for eta-slice CCDF")
    parser.add_argument("--fixed-eta", type=float, default=1e-4, help="Preferred eta for k-slice CCDF")
    parser.add_argument("--min-points-per-curve", type=int, default=3, help="Minimum laminar samples required to draw a curve")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    lam_df = load_laminar_lengths(args.base_dir)
    stats = pooled_stats(lam_df)
    stats.sort_values(["eta", "k"], inplace=True)
    stats.to_csv(args.stats_out, sep="\t", index=False)

    all_k = np.array(sorted(lam_df["k"].unique()))
    all_eta = np.array(sorted(lam_df["eta"].unique()))
    fixed_k, k_curve_count, k_changed = choose_fixed_value(
        lam_df,
        fixed_col="k",
        preferred_value=args.fixed_k,
        varying_col="eta",
        min_points_per_curve=args.min_points_per_curve,
    )
    fixed_eta, eta_curve_count, eta_changed = choose_fixed_value(
        lam_df,
        fixed_col="eta",
        preferred_value=args.fixed_eta,
        varying_col="k",
        min_points_per_curve=args.min_points_per_curve,
    )

    n_eta_slice = make_slice_plot(
        lam_df,
        fixed_col="k",
        fixed_value=fixed_k,
        varying_col="eta",
        out_path=os.path.join(args.plots_dir, "laminar_ccdf_eta_slice.png"),
        min_points_per_curve=args.min_points_per_curve,
    )
    n_k_slice = make_slice_plot(
        lam_df,
        fixed_col="eta",
        fixed_value=fixed_eta,
        varying_col="k",
        out_path=os.path.join(args.plots_dir, "laminar_ccdf_k_slice.png"),
        min_points_per_curve=args.min_points_per_curve,
    )

    heatmap_from_stats(
        stats,
        value_col="laminar_mean",
        title="Mean Laminar Length",
        out_path=os.path.join(args.plots_dir, "laminar_mean_heatmap.png"),
    )
    heatmap_from_stats(
        stats,
        value_col="laminar_p90",
        title="Laminar Length 90th Percentile",
        out_path=os.path.join(args.plots_dir, "laminar_p90_heatmap.png"),
    )

    print(f"Wrote stats: {args.stats_out}")
    print(f"Wrote plots in: {args.plots_dir}")
    if k_changed:
        print(f"Adjusted fixed-k from {args.fixed_k} to {fixed_k} to show more eta curves ({k_curve_count} valid groups)")
    if eta_changed:
        print(f"Adjusted fixed-eta from {args.fixed_eta} to {fixed_eta} to show more k curves ({eta_curve_count} valid groups)")
    print(f"Drawn curves: eta-slice={n_eta_slice}, k-slice={n_k_slice}, min-points-per-curve={args.min_points_per_curve}")


if __name__ == "__main__":
    main()
