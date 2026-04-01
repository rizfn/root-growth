import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    for col in ["delta", "t_mean_escaped", "t_p25_escaped", "t_p75_escaped"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    samples = samples.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta", "t_esc_tau"])
    summary = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta"])
    return samples, summary


def add_panel(ax: plt.Axes, samples: pd.DataFrame, summary: pd.DataFrame, title: str, x_label: str, censored_level: float) -> None:
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
            label="individual escaped",
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
            label="mean escaped",
            zorder=4,
        )

    band_ok = summary.dropna(subset=["t_p25_escaped", "t_p75_escaped"]).sort_values("delta")
    if len(band_ok):
        ax.fill_between(
            band_ok["delta"],
            band_ok["t_p25_escaped"],
            band_ok["t_p75_escaped"],
            color="#D6604D",
            alpha=0.18,
            linewidth=0,
            label="IQR escaped",
            zorder=3,
        )

    # Reference slope: T ~ C / sqrt(delta)
    fit_df = summary[(summary["n_escaped"] >= 3) & np.isfinite(summary["t_mean_escaped"])].copy()
    fit_df = fit_df[(fit_df["delta"] > 0) & (fit_df["t_mean_escaped"] > 0)]
    if len(fit_df) >= 3:
        x = fit_df["delta"].values
        y = fit_df["t_mean_escaped"].values
        log_c = float(np.mean(np.log(y) + 0.5 * np.log(x)))
        c_ref = np.exp(log_c)
        x_line = np.geomspace(x.min() * 0.8, x.max() * 1.2, 250)
        y_line = c_ref * np.power(x_line, -0.5)
        ax.plot(
            x_line,
            y_line,
            color="#4DAC26",
            lw=1.8,
            ls="--",
            label=rf"$C\,\Delta^{{-1/2}}$ ref, $C={c_ref:.1f}$",
            zorder=5,
        )

        slope, _ = np.polyfit(np.log(x), np.log(y), 1)
        ax.annotate(
            rf"measured slope: ${slope:.2f}$",
            xy=(0.04, 0.08),
            xycoords="axes fraction",
            fontsize=11,
            color="#2166AC",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2166AC", alpha=0.85),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$T_{esc}/\tau$")
    ax.set_title(title)
    style_axes(ax)


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
        title="(a) LC2 onset ghost (arrival side)",
        x_label=r"$\Delta_{arr}=k_c-k$",
        censored_level=0.85 * t_max_arr,
    )

    add_panel(
        ax2,
        dis_samples,
        dis_summary,
        title="(b) Chaotic LC2 ghost (disappearance side)",
        x_label=r"$\Delta_{dis}=k-k_c$",
        censored_level=0.85 * t_max_dis,
    )

    # Keep panel-specific legends compact and consistent.
    ax1.legend(loc="upper right", frameon=False)
    ax2.legend(loc="upper right", frameon=False)

    tau_val = float(arr_samples["tau"].iloc[0]) if len(arr_samples) else 1.0
    dt_val = float(arr_samples["dt"].iloc[0]) if len(arr_samples) else 0.01
    fig.suptitle(
        rf"LC2 ghost escape times ($\tau={tau_val:g}$, $dt={dt_val:g}$, deterministic DDE)",
        y=1.02,
        fontsize=22,
    )

    out_png = os.path.join(PLOT_DIR, "lc2_ghost_escape_times.png")
    out_pdf = os.path.join(PLOT_DIR, "lc2_ghost_escape_times.pdf")

    fig.savefig(out_png, format="png", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(out_pdf, format="pdf", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
