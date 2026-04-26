import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import signal
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "SDDETimeseries" / "tau_k_raster"
PLOT_DIR = SCRIPT_DIR / "plots" / "autocorrelation_damping"

TAU = 1.0
THETA0 = 1.5708
# Integration dt is set in the solver; analysis uses sampled dt inferred from files.
PREFERRED_RECORD_DT = None
TMAX = 1000.0
K_VALUES = (1.6, 2.0, 4.0, 4.7)
MU_VALUES = (
    0.0,
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.0007,
    0.001,
    0.002,
    0.003,
    0.005,
    0.007,
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1,
)
# Show only decade noise levels in the autocorrelation panel.
MU_VALUES_ACF = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
TRANSIENT_FRAC = 0.2
MAX_LAG_TIME = 240.0

RC_PARAMS = {
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


def set_plot_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.3, linewidth=0.7)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False, pad=1)


def folder_name(tau: float, k: float, theta0: float, dt: float, tmax: float) -> str:
    return f"tau_{tau:g}_k_{k:g}_theta0_{theta0:g}_dt_{dt:g}_tmax_{tmax:g}"


def parse_folder_params(folder_name_str: str) -> dict[str, float] | None:
    pattern = (
        r"^tau_(?P<tau>[-+0-9.eE]+)_k_(?P<k>[-+0-9.eE]+)_theta0_(?P<theta0>[-+0-9.eE]+)"
        r"_dt_(?P<dt>[-+0-9.eE]+)_tmax_(?P<tmax>[-+0-9.eE]+)$"
    )
    match = re.match(pattern, folder_name_str)
    if not match:
        return None

    try:
        return {
            "tau": float(match.group("tau")),
            "k": float(match.group("k")),
            "theta0": float(match.group("theta0")),
            "dt": float(match.group("dt")),
            "tmax": float(match.group("tmax")),
        }
    except ValueError:
        return None


def approx_equal(a: float, b: float, rel: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1.0))


def find_source_folder(k: float) -> Path | None:
    if not OUTPUT_ROOT.exists():
        return None

    matches = []
    for folder in OUTPUT_ROOT.iterdir():
        if not folder.is_dir():
            continue
        params = parse_folder_params(folder.name)
        if params is None:
            continue
        if not approx_equal(params["tau"], TAU, rel=1e-6):
            continue
        if not approx_equal(params["k"], k, rel=1e-6):
            continue
        score = abs(params["theta0"] - THETA0) + abs(params["tmax"] - TMAX)
        if PREFERRED_RECORD_DT is not None:
            score += abs(params["dt"] - PREFERRED_RECORD_DT)

        # Prefer newer folders when parameter scores are tied.
        try:
            mtime = folder.stat().st_mtime
        except OSError:
            mtime = 0.0
        matches.append((score, -mtime, folder))

    if not matches:
        return None

    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[0][2]


def parse_mu(file_path: Path) -> float | None:
    match = re.search(r"eta_([^_]+)_simNo_", str(file_path))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def discover_mu_values(folder_path: Path) -> list[float]:
    mu_values = set()
    for file_path in folder_path.glob("eta_*_simNo_*.tsv"):
        mu = parse_mu(file_path)
        if mu is not None:
            mu_values.add(mu)
    return sorted(mu_values)


def load_timeseries(file_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        data = np.loadtxt(file_path, skiprows=1)
    except Exception:
        return None

    if data.ndim == 1:
        if data.size < 2:
            return None
        data = data.reshape(1, -1)

    if data.shape[0] < 10 or data.shape[1] < 2:
        return None

    return data[:, 0], data[:, 1]


def infer_sample_dt(time: np.ndarray) -> float | None:
    if len(time) < 2:
        return None
    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


def compute_autocorrelation(time_series: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(time_series)
    if n < 2:
        return np.ones(max(1, max_lag))

    centered = time_series - np.mean(time_series)
    var = np.var(centered)
    if var == 0:
        return np.ones(max_lag)

    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[n - 1 : n - 1 + max_lag].astype(float)
    counts = np.arange(n, n - len(autocorr), -1, dtype=float)
    autocorr /= var * counts
    return autocorr


def fit_damped_oscillation(lag_times: np.ndarray, autocorr: np.ndarray) -> float | None:
    if len(lag_times) < 6 or len(autocorr) < 6:
        return None

    step = float(lag_times[1] - lag_times[0])
    if step <= 0:
        return None

    centered = autocorr - np.mean(autocorr)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(autocorr), d=step)
    if len(freqs) < 2:
        return None

    peak_idx = int(np.argmax(spectrum[1:]) + 1)
    omega_guess = 2.0 * np.pi * float(freqs[peak_idx])

    envelope = np.abs(signal.hilbert(autocorr))
    valid_idx = envelope > 0.01
    lambda_guess = 0.01
    if np.count_nonzero(valid_idx) > 10:
        try:
            popt_env, _ = curve_fit(
                lambda t, amp, lam: amp * np.exp(-lam * t),
                lag_times[valid_idx],
                envelope[valid_idx],
                p0=[1.0, 0.01],
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=4000,
            )
            lambda_guess = float(popt_env[1])
        except Exception:
            lambda_guess = 0.01

    def damped_osc(t: np.ndarray, amp: float, lam: float, omega: float, phase: float) -> np.ndarray:
        return amp * np.exp(-lam * t) * np.cos(omega * t + phase)

    try:
        popt, _ = curve_fit(
            damped_osc,
            lag_times,
            autocorr,
            p0=[1.0, lambda_guess, omega_guess, 0.0],
            bounds=([0.0, 0.0, 0.0, -2.0 * np.pi], [np.inf, np.inf, np.inf, 2.0 * np.pi]),
            maxfev=10000,
        )
    except Exception:
        return None

    return float(popt[1])


def collect_ensemble(folder_path: Path, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    sim_files = []
    for file_path in sorted(folder_path.glob("eta_*_simNo_*.tsv")):
        parsed_mu = parse_mu(file_path)
        if parsed_mu is not None and approx_equal(parsed_mu, mu, rel=1e-9, abs_tol=1e-12):
            sim_files.append(file_path)
    if not sim_files:
        return None

    autocorrs = []
    damping_rates = []
    dt_ref = None

    for sim_file in sim_files:
        loaded = load_timeseries(sim_file)
        if loaded is None:
            continue

        time, theta = loaded
        transient_idx = int(TRANSIENT_FRAC * len(theta))
        theta_steady = theta[transient_idx:]
        time_steady = time[transient_idx:]
        if len(theta_steady) < 10:
            continue

        dt_sample = infer_sample_dt(time_steady)
        if dt_sample is None:
            continue

        if dt_ref is None:
            dt_ref = dt_sample
        elif not approx_equal(dt_sample, dt_ref, rel=1e-6, abs_tol=1e-9):
            # Keep ensemble consistent in lag-time units.
            continue

        max_lag = min(int(MAX_LAG_TIME / dt_ref), len(theta_steady) // 2)
        if max_lag < 6:
            continue

        autocorr = compute_autocorrelation(theta_steady, max_lag)
        autocorrs.append(autocorr)

        lag_times = np.arange(len(autocorr), dtype=float) * dt_ref
        lam = fit_damped_oscillation(lag_times, autocorr)
        if lam is not None and np.isfinite(lam):
            damping_rates.append(lam)

    if not autocorrs:
        return None

    min_len = min(len(arr) for arr in autocorrs)
    autocorr_stack = np.vstack([arr[:min_len] for arr in autocorrs])
    autocorr_mean = np.mean(autocorr_stack, axis=0)
    autocorr_sem = np.std(autocorr_stack, axis=0) / np.sqrt(len(autocorr_stack))
    if dt_ref is None:
        return None
    lag_times = np.arange(min_len, dtype=float) * dt_ref

    damping_array = np.array(damping_rates, dtype=float)
    damping_array = damping_array[np.isfinite(damping_array) & (damping_array > 0)]
    if len(damping_array) > 0:
        damping_mean = np.array([float(np.mean(damping_array))])
        damping_sem = np.array([float(np.std(damping_array) / np.sqrt(len(damping_array)))])
    else:
        damping_mean = np.array([])
        damping_sem = np.array([])

    return lag_times, autocorr_mean, autocorr_sem, damping_mean, damping_sem


def plot_for_k(k: float) -> None:
    folder_path = find_source_folder(k)
    if folder_path is None:
        print(f"Folder not found for k={k:g}")
        return

    available_mu = set(discover_mu_values(folder_path))
    mu_values = [mu for mu in MU_VALUES if mu in available_mu]
    if not mu_values:
        print(f"No simulation files found in {folder_path}")
        return

    print(f"Using source folder: {folder_path}")
    print(f"Found {len(mu_values)} noise values: {[f'{mu:g}' for mu in mu_values]}")

    positive_mu = [mu for mu in mu_values if mu > 0]
    mu_min = min(positive_mu) if positive_mu else 1e-4
    mu_max = max(positive_mu) if positive_mu else 1.0

    fig, (ax_acf, ax_damp) = plt.subplots(1, 2, figsize=(16.5, 6.6), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    cmap = plt.cm.viridis
    norm = LogNorm(vmin=mu_min, vmax=mu_max) if positive_mu else None
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap) if norm is not None else None
    if scalar_mappable is not None:
        scalar_mappable.set_array([])

    zero_mu_plotted = False
    acf_handles = []
    acf_labels = []
    damping_x = []
    damping_y = []
    damping_yerr = []

    for mu in mu_values:
        ensemble = collect_ensemble(folder_path, mu)
        if ensemble is None:
            continue

        lag_times, autocorr_mean, autocorr_sem, damping_mean, damping_sem = ensemble

        if mu == 0.0 or any(approx_equal(mu, mu_keep, rel=1e-10, abs_tol=1e-12) for mu_keep in MU_VALUES_ACF[1:]):
            if mu == 0.0:
                (line,) = ax_acf.plot(lag_times, autocorr_mean, color="#444444", lw=1.9, label=r"$\mu=0$")
                ax_acf.fill_between(lag_times, autocorr_mean - autocorr_sem, autocorr_mean + autocorr_sem, color="#444444", alpha=0.14)
                zero_mu_plotted = True
                acf_handles.append(line)
                acf_labels.append(r"$\mu=0$")
            else:
                color = cmap(norm(mu)) if norm is not None else cmap(0.5)
                mu_exp = int(np.round(np.log10(mu)))
                label = rf"$\mu=10^{{{mu_exp}}}$"
                (line,) = ax_acf.plot(lag_times, autocorr_mean, color=color, lw=1.5, label=label)
                ax_acf.fill_between(lag_times, autocorr_mean - autocorr_sem, autocorr_mean + autocorr_sem, color=color, alpha=0.14)
                acf_handles.append(line)
                acf_labels.append(label)

        if mu > 0 and len(damping_mean) > 0:
            damping_x.append(mu)
            damping_y.append(float(damping_mean[0]))
            damping_yerr.append(float(damping_sem[0]))

    if damping_x:
        damping_x_arr = np.array(damping_x, dtype=float)
        damping_y_arr = np.array(damping_y, dtype=float)
        damping_yerr_arr = np.array(damping_yerr, dtype=float)
        order = np.argsort(damping_x_arr)
        damping_x_arr = damping_x_arr[order]
        damping_y_arr = damping_y_arr[order]
        damping_yerr_arr = damping_yerr_arr[order]

        if norm is not None:
            colors = cmap(norm(damping_x_arr))
        else:
            colors = ["#901A1E"] * len(damping_x_arr)

        for mu, lam, lam_err, color in zip(damping_x_arr, damping_y_arr, damping_yerr_arr, colors):
            ax_damp.errorbar(mu, lam, yerr=lam_err, fmt="o", markersize=6.5, color=color, capsize=4, zorder=3)

    if scalar_mappable is not None:
        cbar = fig.colorbar(scalar_mappable, ax=[ax_acf, ax_damp], pad=0.02, fraction=0.03)
        cbar.set_label(r"$\mu$")

    ax_acf.set_xlabel(r"Lag time")
    ax_acf.set_ylabel(r"Autocorrelation")
    ax_acf.set_xlim(0.0, 20)
    ax_acf.set_title(r"Ensemble-averaged autocorrelation")
    ax_acf.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    if acf_handles:
        ax_acf.legend(acf_handles, acf_labels, loc="upper right", frameon=False)
    style_axes(ax_acf)

    ax_damp.set_xlabel(r"Noise strength $\mu$")
    ax_damp.set_ylabel(r"Damping rate $\lambda$")
    ax_damp.set_title(r"Autocorrelation damping rate")
    ax_damp.set_xscale("log")
    ax_damp.set_yscale("log")
    style_axes(ax_damp)

    fig.suptitle(rf"Autocorrelation damping ($\tau={TAU:g}$, $k={k:g}$)", y=1.02, fontsize=22)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOT_DIR / f"autocorr_damping_tau_{TAU:g}_k_{k:g}.png"
    out_pdf = PLOT_DIR / f"autocorr_damping_tau_{TAU:g}_k_{k:g}.pdf"
    fig.savefig(out_png, format="png", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(out_pdf, format="pdf", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    set_plot_style()
    for k in K_VALUES:
        plot_for_k(k)


if __name__ == "__main__":
    main()
