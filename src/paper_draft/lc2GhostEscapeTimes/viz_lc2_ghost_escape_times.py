import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

ARRIVAL_SAMPLES = os.path.join(SCRIPT_DIR, "outputs", "arrival", "arrival_samples.tsv")
ARRIVAL_SUMMARY = os.path.join(SCRIPT_DIR, "outputs", "arrival", "arrival_summary.tsv")
DISAPP_SAMPLES = os.path.join(SCRIPT_DIR, "outputs", "disappearance", "disappearance_samples.tsv")
DISAPP_SUMMARY = os.path.join(SCRIPT_DIR, "outputs", "disappearance", "disappearance_summary.tsv")


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 28,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
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


def load_tables(samples_path: str, summary_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    samples = pd.read_csv(samples_path, sep="\t")
    summary = pd.read_csv(summary_path, sep="\t")

    for col in ["delta", "t_esc_tau"]:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")
    for col in ["delta", "t_mean_escaped"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    samples = samples.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta", "t_esc_tau"])
    summary = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta"])
    return samples, summary


def add_panel(
    ax: plt.Axes,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    x_label: str,
    censored_level: float,
    add_fitted_power_law: bool = False,
) -> None:
    esc = samples[samples["censored"] == 0]
    cen = samples[samples["censored"] == 1]

    if len(esc):
        ax.scatter(
            esc["delta"],
            esc["t_esc_tau"],
            s=11,
            color="#2166AC",
            alpha=0.16,
            linewidths=0,
            label="Simulation data",
            zorder=2,
        )

    if len(cen):
        ax.scatter(
            cen["delta"],
            np.full(len(cen), censored_level),
            s=14,
            marker="v",
            color="#2166AC",
            alpha=0.22,
            linewidths=0,
            label="censored",
            zorder=2,
        )

    mean_ok = summary.dropna(subset=["t_mean_escaped"]).sort_values("delta")
    if len(mean_ok):
        ax.plot(
            mean_ok["delta"],
            mean_ok["t_mean_escaped"],
            color="#D6604D",
            lw=2.0,
            label="Mean escape time",
            zorder=4,
        )

    # Reference slope: T ~ C / sqrt(delta)
    fit_df = summary[(summary["n_escaped"] >= 3) & np.isfinite(summary["t_mean_escaped"])].copy()
    fit_df = fit_df[(fit_df["delta"] > 0) & (fit_df["t_mean_escaped"] > 0)]
    if len(fit_df) >= 3:
        x = fit_df["delta"].values
        # Fixed reference C and slope gamma=1/2
        C_ref = 10.0
        x_line = np.geomspace(x.min() * 0.8, x.max() * 1.2, 250)
        y_line = C_ref * np.power(x_line, -0.5)
        ax.plot(
            x_line,
            y_line,
            color="#666666",
            lw=1.8,
            ls="--",
            label=r"$\gamma=1/2$ power law",
            zorder=5,
        )

        if add_fitted_power_law:
            log_x = np.log(x)
            log_y = np.log(fit_df["t_mean_escaped"].values)
            slope, _intercept = np.polyfit(log_x, log_y, 1)
            gamma_fit = float(-slope)
            y_fit = C_ref * np.power(x_line, -gamma_fit)
            ax.plot(
                x_line,
                y_fit,
                color="#CBA810",
                lw=1.8,
                ls="-.",
                label=rf"$\gamma={gamma_fit:.2f}$ power law",
                zorder=6,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$T_{esc}/\tau$")
    style_axes(ax)


def add_panel_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    dark_data_marker = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=6,
        markerfacecolor="#2166AC",
        markeredgecolor="#2166AC",
        alpha=0.9,
    )
    dark_censored_marker = Line2D(
        [0],
        [0],
        marker="v",
        linestyle="None",
        markersize=6,
        markerfacecolor="#2166AC",
        markeredgecolor="#2166AC",
        alpha=0.9,
    )

    remapped_handles: list[object] = []
    for h, lbl in zip(handles, labels):
        if lbl == "Simulation data":
            remapped_handles.append(dark_data_marker)
        elif lbl == "censored":
            remapped_handles.append(dark_censored_marker)
        else:
            remapped_handles.append(h)

    ax.legend(remapped_handles, labels, loc="upper right", frameon=False)


def main() -> None:
    set_plot_style()
    os.makedirs(PLOT_DIR, exist_ok=True)

    arr_samples, arr_summary = load_tables(ARRIVAL_SAMPLES, ARRIVAL_SUMMARY)
    dis_samples, dis_summary = load_tables(DISAPP_SAMPLES, DISAPP_SUMMARY)

    t_max_arr = float(arr_samples["t_max_tau"].iloc[0]) if len(arr_samples) else 1.0
    t_max_dis = float(dis_samples["t_max_tau"].iloc[0]) if len(dis_samples) else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    add_panel(
        ax1,
        arr_samples,
        arr_summary,
        x_label=r"$\Delta_{arr}=k_{c,arr}-k$",
        censored_level=0.85 * t_max_arr,
    )

    add_panel(
        ax2,
        dis_samples,
        dis_summary,
        x_label=r"$\Delta_{dis}=k-k_{c,dis}$",
        censored_level=0.85 * t_max_dis,
        add_fitted_power_law=True,
    )

    # Keep panel-specific legends compact and consistent.
    add_panel_legend(ax1)
    add_panel_legend(ax2)

    out_svg = os.path.join(PLOT_DIR, "lc2_ghost_escape_times.svg")
    out_pdf = os.path.join(PLOT_DIR, "lc2_ghost_escape_times.pdf")

    fig.savefig(out_svg, format="svg", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(out_pdf, format="pdf", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)

    print(f"Saved: {out_svg}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
