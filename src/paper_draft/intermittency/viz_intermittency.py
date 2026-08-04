"""Intermittent route to chaos of the deterministic DDE

    dtheta/dt = -k * sin(theta(t - tau)).

Above tau*k_c ~ 4.988 the period-4*tau orbit is interrupted by chaotic bursts that
slide the oscillation centre onto a neighbouring orbit shifted by ~2*pi.  Two
figures are produced:

  1. theta(t) below, just above, and well above the onset, with the laminar
     stretches shaded in alternating colours so the bursts stand out;
  2. the chaotic attractor either side of the crisis, beside the divergence of
     the mean laminar duration at the onset, ell ~ (k*tau - k*tau_c)^(-gamma)
     with gamma ~ 0.82 -- far from the 1/2 of a tangent bifurcation, as expected
     when the onset is a crisis.

The attractor panel is drawn in the difference coordinates

    x = theta(t) - theta(t-tau),   y = theta(t-tau) - theta(t-2*tau),

which are unchanged by theta -> theta + constant.  Since a burst shifts theta by
~2*pi, all the shifted copies of the laminar orbit fall on one closed curve in
these coordinates: the symmetry of the equation does the work that detrending
would otherwise have to.  What comes out is the whole intermittency picture in
one image -- the attractor just below the onset is exactly the sharp core, where
the trajectory sits for the whole of a laminar phase, and the filaments around it
are the territory that only opened up at the crisis, which a burst is one
excursion into and back.

k*tau_c is not read off the scan but fitted jointly with gamma, since a power law
is only straight in log-log for the right onset.  The minimum is interior and
sharp: it moves by <1e-4 whether the fit uses the points with >=100, >=1000 or
>=3000 completed laminar phases, gamma stays within 0.81-0.82, and the residuals
about the fitted line are unstructured across all 3.5 decades of eps.  Those
residuals are far larger than the statistical error bars: the periodic windows
that are dense above the onset shift individual k off the line, and that scatter,
not the counting statistics, is what limits the exponent.

Sporadic bursting is seen a little below the fitted onset (from k*tau ~ 4.9875,
while nothing bursts at all up to 4.9871).  Those trajectories burst a few times
and then lock onto one of the periodic windows and never burst again, i.e. below
the crisis the intermittent set is a chaotic saddle rather than an attractor;
runs with fewer than MIN_EVENTS_FIT completed laminar phases are dominated by
that transient and are left out of the fit.

tau = 1 throughout, so times are quoted bare rather than as t/tau.

Inputs come from runTimeseries.sh, runAttractor.sh and runLaminarScan.sh.
"""

import colorsys
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESERIES_DIR = os.path.join(SCRIPT_DIR, "outputs", "timeseries")
SCAN_DIR = os.path.join(SCRIPT_DIR, "outputs", "laminar_scan")
ATTRACTOR_DIR = os.path.join(SCRIPT_DIR, "outputs", "attractor")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

TAU = 1.0
TWO_PI = 2.0 * np.pi
PERIOD = 4.0 * TAU

# Laminar/burst detector, matching laminar_lengths.cpp: the oscillation centre
# leaves its plateau when it has moved by more than half the ~2*pi spacing to the
# next one, and the burst is over once the centre has been quiet for a period.
SLIP_THRESHOLD = np.pi
QUIET_THRESHOLD = 0.5

# A k contributes to the fit once it has this many completed laminar phases;
# below it the sample is the handful of bursts a trajectory manages before it
# locks onto a periodic window (see module docstring).
MIN_EVENTS_FIT = 100

BOOTSTRAP_SAMPLES = 400

# Laminar phases shorter than this are the momentary pauses of the oscillation
# centre inside a burst, not laminar phases of the period-4*tau orbit; the
# characteristic laminar time is measured from the excess above the cutoff.
# Above it the durations are exponential: the estimate below is flat in the
# cutoff (100, 200 and 400 agree to a few percent).
ELL_MIN = 100.0

# The three regimes, shown in both figures.
EXAMPLE_TAUKS = [4.9700, 5.0000, 5.2000]
EXAMPLE_LABELS = [r"$k\tau < k\tau_c$", r"$k\tau \gtrsim k\tau_c$", r"$k\tau > k\tau_c$"]

TRACE_COLOR = "#901A1E"
# Alternating backgrounds for successive laminar stretches.  Stretches shorter
# than a few periods are the pauses of the centre inside a burst; shading them
# would turn the burst into a stripe pattern, so they are left blank like the
# rest of the burst.
SHADE_COLORS = ["#2166AC", "#CBA810"]
SHADE_LIGHTNESS = 0.88
SHADE_ALPHA = 0.55
SHADE_MIN_DURATION = 5.0 * PERIOD

DATA_COLOR = "#901A1E"
FIT_COLOR = "#2166AC"
REFERENCE_COLOR = "#666666"

# Attractor panel: the density ramp from the sparsest filaments to the laminar
# core, the colour of the pre-crisis attractor drawn over it, and the resolution
# of the density grid.
ATTRACTOR_COLORS = ["#DCE9F3", "#92C5DE", "#4393C3", "#2166AC", "#123A5C"]
PRE_CRISIS_COLOR = "#901A1E"
ATTRACTOR_BINS = 1100
# The pre-crisis attractor is drawn as a curve rather than a density, so only as
# much of it as fills its own width is needed -- a few hundred turns round the
# orbit.  Drawing the whole run overflows Agg's limit on vertices in one path.
PRE_CRISIS_POINTS = 100_000


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 26,
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


def label_extreme_ticks_only(ax: plt.Axes, which: str) -> None:
    """Keep only the outermost tick labels, so an inward axis label has room.

    Call after the limits are final.  The ticks are frozen at the positions the
    automatic locator picked, otherwise constrained_layout re-runs the locator
    while drawing and the labels no longer land on the ticks they were made for.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    low, high = ax.get_xlim() if which == "x" else ax.get_ylim()
    ticks = [t for t in axis.get_majorticklocs() if low <= t <= high]
    if len(ticks) < 2:
        return

    base = axis.get_major_formatter()
    base.set_locs(ticks)  # a ScalarFormatter formats to '' until it knows its ticks
    labels = [base(t, i) for i, t in enumerate(ticks)]
    labels[1:-1] = [""] * (len(labels) - 2)
    axis.set_ticks(ticks)
    axis.set_ticklabels(labels)


def pull_labels_inward(ax: plt.Axes, x: float | None = None, y: float | None = None) -> None:
    """Move the axis labels onto the axes, into the gap left by the thinned ticks.

    Coordinates are axes fractions, so panels keep the same label position however
    wide their tick labels are.
    """
    if y is not None:
        ax.yaxis.set_label_coords(y, 0.5)
    if x is not None:
        ax.xaxis.set_label_coords(0.5, x)


def lighten(color: str, lightness: float = SHADE_LIGHTNESS) -> tuple[float, float, float]:
    """Same hue and saturation, raised lightness -- a background tint of `color`."""
    hue, _, saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def save_figure(fig: plt.Figure, name: str, dpi: float | None = None) -> None:
    """Save as svg; dpi applies only to rasterised elements, text stays vector."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    for ext in ("svg",):
        path = os.path.join(PLOT_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext, dpi=dpi, transparent=True, facecolor="none",
                    edgecolor="none")
        print(f"Saved: {path}")
    plt.close(fig)


def load_timeseries(tauk: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(TIMESERIES_DIR, f"tauk_{tauk:.4f}.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing timeseries: {path}  (run runTimeseries.sh)")
    data = np.loadtxt(path, skiprows=1)
    t, theta, centre = data[:, 0], data[:, 1], data[:, 2]
    # theta -> theta - 2*pi*m is again a solution, so shifting every panel to start
    # near zero compares like with like.
    shift = TWO_PI * np.round(centre[0] / TWO_PI)
    return t, theta - shift, centre - shift


def load_scan() -> dict[float, np.ndarray]:
    files = sorted(glob.glob(os.path.join(SCAN_DIR, "tauk_*.tsv")))
    if not files:
        raise FileNotFoundError(f"No scan data in {SCAN_DIR}  (run runLaminarScan.sh)")

    scan: dict[float, np.ndarray] = {}
    for path in files:
        match = re.search(r"tauk_([0-9.]+)\.tsv$", os.path.basename(path))
        if match is None:
            continue
        data = np.loadtxt(path, skiprows=1, ndmin=2)
        # Columns: run, theta0, t_start, length, censored -> keep length and censored.
        scan[float(match.group(1))] = data[:, 3:5] if data.size else np.empty((0, 2))
    return scan


def load_attractor(tauk: float) -> tuple[np.ndarray, np.ndarray]:
    """The attractor in coordinates that quotient out the 2*pi shifts.

    x = theta(t) - theta(t-tau) and y = theta(t-tau) - theta(t-2*tau) are both
    unchanged by theta -> theta + constant, so the laminar orbit and all of its
    2*pi-shifted copies land on one closed curve, and what surrounds that curve
    is the part of the attractor that only the bursts reach.
    """
    path = os.path.join(ATTRACTOR_DIR, f"tauk_{tauk:.4f}.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing attractor run: {path}  (run runAttractor.sh)")

    data = np.loadtxt(path, skiprows=1)
    theta = data[:, 1]
    lag = int(round(TAU / (data[1, 0] - data[0, 0])))
    return theta[2 * lag:] - theta[lag:-lag], theta[lag:-lag] - theta[:-2 * lag]


def laminar_intervals(t: np.ndarray, centre: np.ndarray) -> list[tuple[float, float]]:
    """Time spans over which the oscillation centre sits on one plateau.

    The same two-state detector as laminar_lengths.cpp, run here on the recorded
    centre so the shading of the timeseries figure and the measured durations mean
    the same thing.
    """
    n_period = int(round(PERIOD / (t[1] - t[0])))

    spans: list[tuple[float, float]] = []
    laminar = True
    centre_lock = centre[0]
    start = t[0]

    for i in range(1, len(t)):
        if laminar:
            if abs(centre[i] - centre_lock) > SLIP_THRESHOLD:
                spans.append((start, t[i]))
                laminar = False
        elif abs(centre[i] - centre[max(0, i - n_period)]) < QUIET_THRESHOLD:
            laminar = True
            centre_lock = centre[i]
            start = t[i]

    if laminar:
        spans.append((start, t[-1]))
    return spans


def laminar_time_scale(lengths: np.ndarray, censored: np.ndarray) -> tuple[float, float, int]:
    """Mean laminar duration above ELL_MIN, from the right-censored exponential MLE.

    A laminar phase that is still running when its trajectory's measurement window
    closes (or when the trajectory locks onto a periodic window and stops bursting)
    contributes its observed duration but not a completed event, so
        ell_bar = sum_i (ell_i - ELL_MIN) / #{completed i},
    which is unbiased for an exponential tail.  Its standard error is
    ell_bar / sqrt(n_events).
    """
    keep = lengths > ELL_MIN
    n_events = int(np.count_nonzero(keep & (censored == 0)))
    if n_events == 0:
        return np.nan, np.nan, 0
    total = float(np.sum(lengths[keep] - ELL_MIN))
    ell_bar = total / n_events
    return ell_bar, ell_bar / np.sqrt(n_events), n_events


def fit_power_law(tauks: np.ndarray, ell: np.ndarray) -> tuple[float, float, float, float]:
    """Fit ell = A * (tau*k - tau*k_c)^(-gamma), with the onset as a free parameter.

    A power law is a straight line in log-log only for the right onset, so tau*k_c
    is chosen to minimise the scatter about that line, and gamma and log A follow
    from the straight-line fit at that onset.
    """
    def scatter(tauk_c: float) -> float:
        eps = tauks - tauk_c
        if eps.min() <= 1e-9:
            return np.inf
        log_eps = np.log(eps)
        slope, intercept = np.polyfit(log_eps, np.log(ell), 1)
        return float(np.sum((np.log(ell) - slope * log_eps - intercept) ** 2))

    result = minimize_scalar(
        scatter,
        bounds=(tauks.min() - 1.0, tauks.min() - 1e-7),
        method="bounded",
        options={"xatol": 1e-10},
    )
    tauk_c = float(result.x)
    log_eps = np.log(tauks - tauk_c)
    slope, intercept = np.polyfit(log_eps, np.log(ell), 1)
    rms = float(np.sqrt(scatter(tauk_c) / len(tauks)))
    return tauk_c, -slope, intercept, rms


def bootstrap_power_law(tauks: np.ndarray, ell: np.ndarray) -> tuple[float, float]:
    """Standard errors on the onset and the exponent, by resampling the k."""
    rng = np.random.default_rng(0)
    onsets, gammas = [], []
    for _ in range(BOOTSTRAP_SAMPLES):
        pick = rng.integers(0, len(tauks), len(tauks))
        if len(np.unique(tauks[pick])) < 6:
            continue
        tauk_c, gamma, _, _ = fit_power_law(tauks[pick], ell[pick])
        if tauk_c < tauks[pick].min() - 1e-6:
            onsets.append(tauk_c)
            gammas.append(gamma)
    return float(np.std(onsets)), float(np.std(gammas))


def plot_timeseries() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True, constrained_layout=True)
    fig.patch.set_alpha(0.0)

    shades = [lighten(c) for c in SHADE_COLORS]

    for ax, tauk, label in zip(axes, EXAMPLE_TAUKS, EXAMPLE_LABELS):
        t, theta, centre = load_timeseries(tauk)

        # Alternating tints so two laminar stretches either side of a burst are
        # never the same colour; the gaps between them are the bursts.
        spans = [s for s in laminar_intervals(t, centre)
                 if s[1] - s[0] >= SHADE_MIN_DURATION]
        for i, (start, stop) in enumerate(spans):
            ax.axvspan(start, stop, color=shades[i % len(shades)], alpha=SHADE_ALPHA,
                       lw=0, zorder=0)

        # The fast oscillation is a dense band at this scale; rasterising it keeps
        # the vector file small while the slow centre stays as a curve.
        ax.plot(t, theta, lw=0.3, color=REFERENCE_COLOR, alpha=0.35, zorder=2,
                rasterized=True)
        # The centre varies on the scale of a period, so it stays a light vector
        # curve at a tenth of the recorded resolution.
        stride = max(1, len(t) // 20000)
        ax.plot(t[::stride], centre[::stride], lw=2.4, color=TRACE_COLOR, zorder=3)

        ax.set_ylabel(r"$\theta$")
        # Headroom so the label never sits on top of the trace.
        low, high = ax.get_ylim()
        ax.set_ylim(low, low + 1.25 * (high - low))
        ax.text(
            0.985,
            0.94,
            rf"$k\tau = {tauk:.3f}$   {label}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=24,
            color="black",
        )
        style_axes(ax)
        label_extreme_ticks_only(ax, "y")
        pull_labels_inward(ax, y=-0.028)

    axes[-1].set_xlabel(r"$t$")
    # Round up so the last tick label is the round end of the run, not the sample
    # just short of it.
    axes[-1].set_xlim(0.0, float(np.ceil(load_timeseries(EXAMPLE_TAUKS[0])[0][-1])))
    label_extreme_ticks_only(axes[-1], "x")
    pull_labels_inward(axes[-1], x=-0.055)

    # The dense theta trace is the only rasterised element, and it is a texture:
    # 300 dpi keeps the file small without any visible loss.
    save_figure(fig, f"intermittency_timeseries_tau_{TAU:g}", dpi=300)


def plot_laminar_statistics(scan: dict[float, np.ndarray]) -> None:
    tauks = np.array(sorted(scan))

    ell_bar, ell_err, n_events = [], [], []
    for tauk in tauks:
        data = scan[tauk]
        value, error, count = laminar_time_scale(data[:, 0], data[:, 1])
        ell_bar.append(value)
        ell_err.append(error)
        n_events.append(count)
    ell_bar = np.array(ell_bar)
    ell_err = np.array(ell_err)
    n_events = np.array(n_events)

    ok = n_events >= MIN_EVENTS_FIT
    tauk_c, gamma, intercept, rms = fit_power_law(tauks[ok], ell_bar[ok])
    tauk_c_err, gamma_err = bootstrap_power_law(tauks[ok], ell_bar[ok])
    slope = -gamma
    eps = tauks[ok] - tauk_c
    ell = ell_bar[ok]

    quiet = tauks[n_events == 0]
    bursting = tauks[(n_events > 0) & ~ok]
    print(f"tau*k_c = {tauk_c:.5f} +- {tauk_c_err:.5f} (fitted), "
          f"gamma = {gamma:.3f} +- {gamma_err:.3f}, "
          f"rms log-residual {rms:.2f} over {np.count_nonzero(ok)} points")
    print(f"  no burst at all up to tau*k = {quiet.max():.4f}; "
          f"transient bursting only at {', '.join(f'{k:.4f}' for k in bursting)}")

    for tauk, value, error, count in zip(tauks, ell_bar, ell_err, n_events):
        used = "fit" if count >= MIN_EVENTS_FIT else "   "
        residual = (np.log(value) - slope * np.log(tauk - tauk_c) - intercept
                    if count >= MIN_EVENTS_FIT else np.nan)
        print(f"  tau*k={tauk:.6f}  eps={tauk - tauk_c:+.5f}  "
              f"ell={value:10.1f} +- {error:8.1f}  n={count:6d}  {used}  "
              f"resid={residual:+.2f}")

    # How much the answer depends on where the fit is cut off: the exponent should
    # not care, and the onset should move by much less than the spacing of the k.
    print("  sensitivity to the events cutoff:")
    for cutoff in (MIN_EVENTS_FIT, 1000, 3000):
        window = n_events >= cutoff
        if np.count_nonzero(window) < 6:
            continue
        sub_c, sub_gamma, _, sub_rms = fit_power_law(tauks[window], ell_bar[window])
        print(f"    n >= {cutoff:5d} ({np.count_nonzero(window):2d} points): "
              f"tau*k_c = {sub_c:.5f}, gamma = {sub_gamma:.3f}, rms = {sub_rms:.2f}")

    fig, (ax_attractor, ax_scale) = plt.subplots(1, 2, figsize=(16.0, 7.0),
                                                 constrained_layout=True)
    fig.patch.set_alpha(0.0)

    # ── The attractor, before and after the crisis ───────────────────────────
    # Density of the chaotic attractor just above the onset, with the attractor
    # just below it drawn on top.  The pre-crisis attractor is the sharp core:
    # that is where the trajectory sits during a laminar phase, going round it
    # thousands of times.  Everything outside is territory that only opened up at
    # the crisis, and a burst is one excursion out into it and back.  Chaos here
    # is not a different motion, it is the old periodic motion plus rare trips
    # into the filaments around it.
    x_above, y_above = load_attractor(EXAMPLE_TAUKS[1])
    x_below, y_below = load_attractor(EXAMPLE_TAUKS[0])

    span = 1.02 * max(np.abs(x_above).max(), np.abs(y_above).max())
    density, _, _ = np.histogram2d(x_above, y_above, bins=ATTRACTOR_BINS,
                                   range=[[-span, span], [-span, span]])
    colours = mcolors.LinearSegmentedColormap.from_list("attractor", ATTRACTOR_COLORS)
    colours.set_bad(alpha=0.0)  # empty cells stay transparent, not white
    ax_attractor.imshow(
        np.ma.masked_where(density.T == 0, density.T),
        origin="lower",
        extent=(-span, span, -span, span),
        cmap=colours,
        # Saturating at a high percentile rather than the peak: the core is
        # visited orders of magnitude more often than the filaments, and against
        # its own maximum everything else washes out.
        norm=mcolors.LogNorm(vmin=1.0, vmax=np.percentile(density[density > 0], 99.5)),
        interpolation="nearest",
        zorder=1,
    )
    # ax_attractor.plot(x_below[:PRE_CRISIS_POINTS], y_below[:PRE_CRISIS_POINTS],
    #                   lw=0.7, color=PRE_CRISIS_COLOR, alpha=0.9, zorder=2,
    #                   rasterized=True)

    handles = [
        plt.Line2D([], [], color=ATTRACTOR_COLORS[-2], lw=4,
                   label=rf"$k\tau = {EXAMPLE_TAUKS[1]:.2f}$"),
        # plt.Line2D([], [], color=PRE_CRISIS_COLOR, lw=2.4,
        #            label=rf"$k\tau = {EXAMPLE_TAUKS[0]:.2f}$"),
    ]
    ax_attractor.legend(handles=handles, loc="upper left", frameon=False,
                        handlelength=1.2, borderaxespad=0.2)

    ax_attractor.set_xlim(-span, span)
    ax_attractor.set_ylim(-span, span)
    ax_attractor.set_aspect("equal")
    ax_attractor.set_xlabel(r"$\Delta\theta(t)$")
    ax_attractor.set_ylabel(r"$\Delta\theta(t-\tau)$")
    style_axes(ax_attractor)
    ax_attractor.grid(False)
    label_extreme_ticks_only(ax_attractor, "x")
    label_extreme_ticks_only(ax_attractor, "y")
    pull_labels_inward(ax_attractor, x=-0.06, y=-0.05)

    # ── Divergence of the laminar time scale at the onset ────────────────────
    eps_line = np.geomspace(eps.min() * 0.7, eps.max() * 1.4, 200)
    fit_line = np.exp(intercept) * eps_line ** slope
    ax_scale.plot(eps_line, fit_line, color=FIT_COLOR, lw=2.4, zorder=4,
                  label=rf"$\bar{{\ell}} \sim \epsilon^{{-{gamma:.2f}}}$")
    ax_scale.plot(eps_line, np.exp(intercept) * eps.min() ** (slope + 0.5) * eps_line ** -0.5,
                  color=REFERENCE_COLOR, lw=1.8, ls="--", zorder=2,
                  label=r"$\epsilon^{-1/2}$")
    ax_scale.errorbar(eps, ell, yerr=ell_err[ok], fmt="o", markersize=7,
                      color=DATA_COLOR, ecolor=DATA_COLOR, elinewidth=1.2, capsize=3,
                      zorder=5, label="Simulation data")

    ax_scale.set_xscale("log")
    ax_scale.set_yscale("log")
    ax_scale.set_xlabel(r"$\epsilon = k\tau - k\tau_c$")
    ax_scale.set_ylabel(r"$\bar{\ell}$")
    ax_scale.legend(loc="lower left", frameon=False)
    style_axes(ax_scale)
    label_extreme_ticks_only(ax_scale, "x")
    label_extreme_ticks_only(ax_scale, "y")
    pull_labels_inward(ax_scale, x=-0.055, y=-0.045)

    save_figure(fig, f"attractor_scaling_tau_{TAU:g}_taukc_{tauk_c:.5f}", dpi=400)


def main() -> None:
    set_plot_style()
    plot_timeseries()
    plot_laminar_statistics(load_scan())


if __name__ == "__main__":
    main()
