"""Paper figures for the autocorrelation damping (tau = 1).

fig 1  autocorrelation C(Delta) at three noise levels; the damping rate gamma
       against eta; and the prefactor c(k) against k, theory vs numerics.
fig 2  the noise-induced escape from the long cycle to the period-4tau cycle at
       k tau = 4: how the apparent threshold depends on the observation time,
       and the activated rate underneath it.

Colour conventions
    the three accent colours (red / blue / gold) mean k tau in fig 1 and
    observation time in fig 2
    the teal ramp means noise strength, and is used only for C(Delta)

Data live in outputs/damping_analysis.npz, reduced from the C++ solvers here.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "plots", "paper")
DATA = np.load(os.path.join(SCRIPT_DIR, "outputs", "damping_analysis.npz"))

C_RED = "#901A1E"
C_BLUE = "#547AA5"
C_GOLD = "#B3950F"
C_GREY = "#666666"

# noise levels in the C(Delta) panel: a teal ramp, deliberately outside the
# red/blue/gold family used for the other quantities.  The span is kept narrow,
# since the nesting of the envelopes already carries the ordering and a wider
# ramp only costs legibility at both ends.
C_ACF = {"0.05": "#55ABA5", "0.1": "#2E8B85", "0.2": "#16605A"}
# Labels are set in one dark neutral rather than in the curve colours: a ramp
# that separates three curves well is necessarily too pale at one end and too
# dark at the other to read as text.  Each label sits on its own envelope, so
# nothing is lost.  Set to None to colour them by curve instead.
ACF_LABEL_COLOR = "#1A1A1A"

# ---------------------------------------------------------------------------
# hand-tuned label placement, in data coordinates -- move these to taste
# ---------------------------------------------------------------------------
# eta labels in the C(Delta) panel, each sitting in the gap just above its own
# envelope so no label lies on a curve
ACF_LABEL_POSITIONS = {
    "0.05": (154.0, 0.78),
    "0.1": (154.0, 0.5),
    "0.2": (154.0, 0.0),
}
ETA2_LABEL_POSITION = (1e-3, 1e-4)   # "propto eta^2" in the gamma(eta) panel
# k tau labels in the gamma(eta) panel: colour-coded text instead of a legend,
# so nothing in the panel can be mistaken for a data point
KTAU_LABEL_POSITIONS = {1.8: (2.2e-2, 2.5e-8),
                        2.0: (2.2e-2, 3.0e-9),
                        3.0: (2.2e-2, 3.6e-10)}
# same treatment for the observation times in fig 2
TOBS_LABEL_POSITIONS = {1e3: (3.1e-2, 0.60),
                        1e4: (3.1e-2, 0.42),
                        1e5: (3.1e-2, 0.24)}
ESC_LABEL_POSITION = (5000.0, 4.0e4)   # scaling label, just above the fitted line

# axis-label placement in AXES coordinates: x for y-labels, y for x-labels.
# More negative pushes the label away from the frame.  Using axes coordinates
# rather than labelpad keeps the placement independent of how wide the tick
# labels happen to be.
XLAB_Y = -0.05
YLAB_X = {"acf": -0.10, "gamma": -0.145, "c": -0.155,
          "frac": -0.105, "lnr": -0.135}

FIG2_FONT_SCALE = 1.0
LAG_MAX = 200.0
K_PANEL = [(1.8, C_BLUE), (2.0, C_GOLD), (3.0, C_RED)]
# above this eta the cycle counter mis-counts for the small near-Hopf cycles
ETA_CAP = {1.8: 0.1}


def set_style(scale=1.0):
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 30 * scale,
        "axes.labelsize": 34 * scale,
        "xtick.labelsize": 29 * scale,
        "ytick.labelsize": 29 * scale,
        "legend.fontsize": 28 * scale,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def finish(ax, which="major"):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.minorticks_on()
    ax.grid(False, which="both")
    ax.grid(True, which=which, alpha=0.3)
    ax.tick_params(axis="both", which="both", direction="out",
                   top=False, right=False, pad=2)
    # log axes spanning less than a decade otherwise label their minor ticks
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def label_only(ax, axis, targets, labels):
    """Keep every tick mark, but write a label only at the given positions."""
    obj = ax.xaxis if axis == "x" else ax.yaxis
    lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ticks = obj.get_ticklocs()
    visible = ticks[(ticks >= lo) & (ticks <= hi)]
    inview = [t for t in targets if lo <= t <= hi]
    allt = np.array(sorted(set(np.concatenate((visible, np.array(inview, dtype=float))))))
    obj.set_major_locator(FixedLocator(allt))
    tol = 1e-9 * max(1.0, *(abs(t) for t in targets))

    def fmt(val, _):
        for t, lab in zip(targets, labels):
            if np.isclose(val, t, rtol=0.0, atol=tol):
                return lab
        return ""

    obj.set_major_formatter(FuncFormatter(fmt))


def log_ticks(ax, axis, ticks, labelled):
    """Ticks at the given decades, labels only at `labelled` (kept near the axis
    ends so the centred axis label has room)."""
    obj = ax.xaxis if axis == "x" else ax.yaxis
    obj.set_major_locator(FixedLocator(ticks))

    def fmt(val, _):
        for t in labelled:
            if np.isclose(np.log10(val), np.log10(t), atol=1e-9):
                return rf"$10^{{{int(round(np.log10(t)))}}}$"
        return ""

    obj.set_major_formatter(FuncFormatter(fmt))


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), format=ext,
                    transparent=True, facecolor="none", edgecolor="none",
                    bbox_inches="tight")
    print("Saved:", os.path.join(OUT_DIR, name))


def place_labels(ax, ylab_x, xlab_y=None):
    ax.yaxis.set_label_coords(ylab_x, 0.5)
    ax.xaxis.set_label_coords(0.5, XLAB_Y if xlab_y is None else xlab_y)


def gamma_series(k):
    """gamma(eta) at one k, from the phase-diffusion coefficient."""
    D_k, D_eta, D = DATA["D_k"], DATA["D_eta"], DATA["D"]
    i = int(np.argmin(np.abs(D_k - k)))
    assert abs(D_k[i] - k) < 1e-9, f"no phase-diffusion data at k={k}"
    cap = ETA_CAP.get(k, np.inf)
    sel = (D_eta > 0) & (D_eta <= cap) & np.isfinite(D[i]) & (D[i] > 0)
    return D_eta[sel], D[i][sel]


# =============================================================================
# Figure 1
# =============================================================================
def figure_damping():
    set_style()
    fig = plt.figure(figsize=(18.5, 5.2))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    fig.patch.set_alpha(0.0)

    # ---- autocorrelation ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    lag = DATA["disp_lag"]
    m = lag <= LAG_MAX
    for es, col in C_ACF.items():
        env = DATA[f"disp_env_eta{es}"][m]
        ax.fill_between(lag[m], -env, env, color=col, alpha=0.16, lw=0)
    for es, col in C_ACF.items():
        ax.plot(lag[m], DATA[f"disp_acf_eta{es}"][m], lw=0.45, color=col, alpha=0.38)
    # outline the envelopes: with three overlapping bands the fills alone do not
    # separate the noise levels
    for es, col in C_ACF.items():
        env = DATA[f"disp_env_eta{es}"][m]
        ax.plot(lag[m], env, lw=2.6, color=col)
        ax.plot(lag[m], -env, lw=2.6, color=col)
        x_text, y_text = ACF_LABEL_POSITIONS[es]
        ax.text(x_text, y_text, rf"$\eta={float(es):g}$",
                color=col if ACF_LABEL_COLOR is None else ACF_LABEL_COLOR,
                fontsize=27, ha="center", va="center")
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.55)
    ax.set_xlim(0.0, LAG_MAX)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$C(\Delta)$")
    finish(ax)
    place_labels(ax, YLAB_X["acf"])
    label_only(ax, "x", (0.0, LAG_MAX), ("0", f"{LAG_MAX:.0f}"))
    label_only(ax, "y", (-1.0, 1.0), ("-1", "1"))

    # ---- gamma against eta --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for k, col in K_PANEL:
        e, g = gamma_series(k)
        ax.loglog(e, g, "o", ms=8, color=col)
        ax.text(*KTAU_LABEL_POSITIONS[k], rf"$k\tau={k:g}$", color=col,
                fontsize=28, ha="center", va="center")
    ee = np.logspace(-4.6, -0.8, 20)
    ax.loglog(ee, 5.0 * ee**2, ls=(0, (7, 5)), lw=2.2, color=C_GREY)
    ax.text(*ETA2_LABEL_POSITION, r"$\propto\eta^{2}$", color=C_GREY,
            fontsize=30, ha="center", va="bottom")
    ax.set_xlim(2e-5, 0.4)
    ax.set_ylim(1e-11, 5e-1)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\gamma$")
    finish(ax)
    place_labels(ax, YLAB_X["gamma"])
    log_ticks(ax, "x", [1e-4, 1e-3, 1e-2, 1e-1], [1e-4, 1e-1])
    log_ticks(ax, "y", [1e-10, 1e-8, 1e-6, 1e-4, 1e-2], [1e-10, 1e-2])

    # ---- prefactor c(k) -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    pk, pc = DATA["prc_k"], DATA["prc_c"]
    sel = pk <= 3.85
    ax.semilogy(pk[sel], pc[sel], "-", lw=2.6, color="black",
                label=r"$\frac{1}{2}\langle Z^{2}\rangle$")
    sel = DATA["cmeas_k"] <= 3.85
    ax.semilogy(DATA["cmeas_k"][sel], DATA["cmeas"][sel], "o", ms=9, mfc="none",
                mew=2.2, color=C_GREY, label="numerics")
    for k, col in K_PANEL:
        j = int(np.argmin(np.abs(DATA["cmeas_k"] - k)))
        ax.semilogy(DATA["cmeas_k"][j], DATA["cmeas"][j], "o", ms=13, color=col)
    ax.axvline(np.pi / 2, ls=(0, (8, 6)), lw=1.2, color=C_GREY, alpha=0.85)
    ax.set_xlim(1.45, 3.95)
    ax.set_ylim(8e-2, 1.4e1)
    ax.set_xlabel(r"$k\tau$")
    ax.set_ylabel(r"$c=\gamma/\eta^{2}$")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.35,
              borderaxespad=0.2, labelspacing=0.15)
    finish(ax)
    place_labels(ax, YLAB_X["c"])
    label_only(ax, "x", (1.5, 3.5), ("1.5", "3.5"))
    log_ticks(ax, "y", [1e-1, 1e0, 1e1], [1e-1, 1e1])

    save(fig, "autocorr_damping_tau_1")
    plt.close(fig)


# =============================================================================
# Figure 2
# =============================================================================
def figure_switching():
    T_OBS = [(1e3, C_BLUE), (1e4, C_GOLD), (1e5, C_RED)]

    # Two panels, not three, so this figure is scaled up more when fitted to
    # \linewidth.  Shrink the type by the same factor so both figures end up
    # with the same apparent size on the page.
    set_style(scale=FIG2_FONT_SCALE)
    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 2, wspace=0.30)
    fig.patch.set_alpha(0.0)

    # ---- fraction switched, for several observation windows -----------------
    ax = fig.add_subplot(gs[0, 0])
    eta = DATA["fs_eta"]
    for Tobs, col in T_OBS:
        j = int(np.argmin(np.abs(DATA["fs_Tobs"] - Tobs)))
        ax.semilogx(eta, DATA["fs_frac"][j], "o-", ms=7, lw=2.0, color=col)
        ax.text(*TOBS_LABEL_POSITIONS[Tobs],
                rf"$T_{{\rm obs}}=10^{{{int(round(np.log10(Tobs)))}}}$",
                color=col, fontsize=24, ha="center", va="center")
    ax.set_xlim(5.5e-3, 5.5e-2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Fraction switched")
    finish(ax)
    place_labels(ax, YLAB_X["frac"])
    log_ticks(ax, "x", [1e-2], [1e-2])
    label_only(ax, "y", (0.0, 1.0), ("0", "1"))

    # ---- the activated escape time ------------------------------------------
    # plotted as the time itself on a log axis rather than ln r on a linear one,
    # so the y axis carries a physical quantity comparable with T_obs at left
    ax = fig.add_subplot(gs[0, 1])
    x, y = DATA["kramers_x"], DATA["kramers_y"]
    A = DATA["kramers_fit"]
    xx = np.linspace(0, 1.06 * x.max(), 50)
    ax.semilogy(xx, np.exp(-np.polyval(A, xx)), ls=(0, (7, 5)), lw=2.0,
                color=C_GREY)
    ax.semilogy(x, np.exp(-y), "o", ms=9, color=C_RED)
    ax.text(*ESC_LABEL_POSITION, r"$\propto e^{\,C/\eta^{2}}$", color=C_GREY,
            fontsize=26, ha="center", va="center")
    ax.set_xlim(0, 1.06 * x.max())
    ax.set_ylim(3e2, 1e7)
    ax.set_xlabel(r"$\eta^{-2}$")
    ax.set_ylabel("Escape time")
    finish(ax)
    place_labels(ax, YLAB_X["lnr"])
    label_only(ax, "x", (0.0, 10000.0), ("0", r"$10^{4}$"))
    log_ticks(ax, "y", [1e3, 1e4, 1e5, 1e6, 1e7], [1e3, 1e7])

    save(fig, "cycle_switching_tau_1_k_4")
    plt.close(fig)
    print(f"  Kramers: ln r = {A[1]:.3f} - {-A[0]:.4e}/eta^2")


if __name__ == "__main__":
    set_style()
    figure_damping()
    figure_switching()
