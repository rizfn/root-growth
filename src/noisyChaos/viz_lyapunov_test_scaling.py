import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_test_summary_map(base_dir: str) -> Dict[str, pd.Series]:
    pattern = os.path.join(base_dir, "outputs", "lyapunov_test", "run_summaries", "*.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No summary files found at {pattern}")

    summary_map: Dict[str, pd.Series] = {}
    for fp in files:
        df = pd.read_csv(fp, sep="\t")
        if df.empty:
            continue
        row = df.iloc[0]
        slug = os.path.splitext(os.path.basename(fp))[0]
        summary_map[slug] = row

    if not summary_map:
        raise RuntimeError("Summary files exist but no rows could be loaded")

    return summary_map


def load_trace_files(base_dir: str) -> List[str]:
    pattern = os.path.join(base_dir, "outputs", "lyapunov_test", "deviation_timeseries", "*.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No deviation timeseries files found at {pattern}")
    return files


def forward_smooth(signal: np.ndarray, dt: float, tau: float) -> np.ndarray:
    window = max(1, int(round((4.0 * tau) / dt)))
    if signal.size < window:
        return np.array([], dtype=float)
    cs = np.cumsum(np.concatenate([[0.0], signal]))
    return (cs[window:] - cs[:-window]) / window


def expected_scaling_from_anchor(anchor: float, lam: float, time: np.ndarray, t0: float) -> np.ndarray:
    return anchor * np.exp(lam * (time - t0))


def make_overlay_plot(
    trace_df: pd.DataFrame,
    summary_row: pd.Series,
    lambda_est: float,
    slug: str,
    out_dir: str,
):
    time = trace_df["time"].to_numpy(dtype=float)
    sep = trace_df["separation"].to_numpy(dtype=float)
    tau = float(summary_row["tau"])

    finite = np.isfinite(time) & np.isfinite(sep) & (sep > 0.0)
    if np.count_nonzero(finite) < 2:
        return np.nan, np.nan

    time = time[finite]
    sep = sep[finite]
    dt = float(np.median(np.diff(time)))

    avg_sep = forward_smooth(sep, dt, tau)
    if avg_sep.size < 4:
        return np.nan, np.nan
    t_avg = time[:avg_sep.size]

    anchor = float(avg_sep[0])
    pred = expected_scaling_from_anchor(anchor, lambda_est, t_avg, t_avg[0])

    fit_count = max(6, int(0.3 * avg_sep.size))
    x_fit = t_avg[:fit_count] - t_avg[0]
    y_fit = np.log(avg_sep[:fit_count])
    slope = float(np.polyfit(x_fit, y_fit, 1)[0])

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.semilogy(time, sep, lw=0.9, alpha=0.35, label="|delta theta|")
    ax.semilogy(t_avg, avg_sep, lw=1.8, alpha=0.95, label="forward avg over 4*tau")
    ax.semilogy(t_avg, pred, lw=2.0, ls="--", label=f"pred: exp(lambda t), lambda={lambda_est:.4g}")

    ax.set_xlabel("time")
    ax.set_ylabel("separation")
    ax.set_title(f"Trace check vs lambda from scan\n{slug}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path = os.path.join(out_dir, f"{slug}.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return slope, float(np.log(anchor))


def build_fit_table(results: List[Tuple[str, float, float]], out_path: str):
    df = pd.DataFrame(
        results,
        columns=["slug", "lambda_from_scan", "slope_from_trace_early"],
    )
    df.sort_values("slug", inplace=True)
    df.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Plot deviation timeseries against exponential scaling from estimated Lyapunov exponent."
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.dirname(__file__),
        help="Directory containing outputs/lyapunov_test",
    )
    parser.add_argument(
        "--plots-dir",
        default=os.path.join(os.path.dirname(__file__), "plots", "lyapunov_test_scaling"),
        help="Output directory for overlay figures",
    )
    parser.add_argument(
        "--fit-table",
        default=os.path.join(os.path.dirname(__file__), "outputs", "lyapunov_test", "scaling_fit_summary.tsv"),
        help="Output TSV for per-test slope checks",
    )
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.fit_table), exist_ok=True)

    test_summary_map = load_test_summary_map(args.base_dir)
    trace_files = load_trace_files(args.base_dir)

    fit_rows = []
    n_written = 0

    for fp in trace_files:
        slug = os.path.splitext(os.path.basename(fp))[0]
        if slug not in test_summary_map:
            continue

        test_row = test_summary_map[slug]
        trace_df = pd.read_csv(fp, sep="\t")
        required = {"time", "separation"}
        if not required.issubset(trace_df.columns):
            continue

        # Read lambda directly from test summary (calculated by lyapunovTest.cpp)
        if "lambda" not in test_row:
            print(f"Skipping {slug}: no lambda in test summary")
            continue
        
        lambda_est = float(test_row["lambda"])
        slope, _ = make_overlay_plot(trace_df, test_row, lambda_est, slug, args.plots_dir)
        fit_rows.append((slug, lambda_est, slope))
        n_written += 1

    build_fit_table(fit_rows, args.fit_table)
    print(f"Wrote {n_written} overlay plots to {args.plots_dir}")
    print(f"Wrote scaling summary table to {args.fit_table}")


if __name__ == "__main__":
    main()
