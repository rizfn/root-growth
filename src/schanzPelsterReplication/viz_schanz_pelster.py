"""
viz_schanz_pelster.py
=====================
Comprehensive verification of Schanz & Pelster (PRE 67, 056205, 2003)
for the zero-noise DDE:
    dθ/dt = -k·sin(θ(t-τ))

Figures produced
----------------
1. bifurcation_coarse.png   – Full bifurcation diagram τ·k ∈ [1.4, 5.5]
                              LOCAL_MAX events, two IC families colour-coded.
2. bifurcation_fine.png     – Period-doubling window τ·k ∈ [4.08, 4.30]
                              Two Poincaré conditions side-by-side:
                                (a) LOCAL_MAX events (θ-values)
                                (b) ZERO_UP events (dθ/dt at upward crossings)
3. coexisting_lc.png        – Timeseries + phase portrait showing the two
                              coexisting limit-cycle branches (IC ±1.0 vs ±2.0)
                              at τ·k ≈ 3.80.
4. period_doubling_ts.png   – Timeseries at τ·k = 4.13→4.20 (pre-PD through chaos)
5. power_spectra.png        – Welch PSD at same τ·k values; subharmonic peaks
                              highlight the PD cascade and chaotic broadening.
6. return_maps.png          – Stroboscopic (Δt = 4τ) return maps at key values.

Run after: bash runBifurcation.sh
"""

import os
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ─────────────────────────────────────────────────────────────────
TAU = 25

LANDMARK_TAUKVALS = {
    r"$\pi/2$"      : np.pi / 2,
    "3.77"          : 3.77,
    "4.105"         : 4.105,
    "4.11"          : 4.11,
    "PD₁ 4.157"     : 4.157,
    "PD₂ 4.165"     : 4.165,
    "PD₃ 4.1725"    : 4.1725,
    "PD₄ 4.17375"   : 4.17375,
    "chaos"         : 4.175,
    "4.24"          : 4.24,
    "4.85"          : 4.85,
    "5.30"          : 5.30,
}

# IC colour mapping: positive/negative pairs share hue, different saturation
IC_PALETTE = {
    0.5  : ("gray", "o"),
    1.0  : ("#2166AC", "."),   # blue family
    -1.0 : ("#92C5DE", "."),   # light blue
    2.0  : ("#D6604D", "."),   # red family
    -2.0 : ("#F4A582", "."),   # salmon
}

IC_LABEL = {
    0.5  : r"IC = +0.5",
    1.0  : r"IC = +1.0  (branch 1+)",
    -1.0 : r"IC = −1.0  (branch 1−)",
    2.0  : r"IC = +2.0  (branch 2+)",
    -2.0 : r"IC = −2.0  (branch 2−)",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIF_DIR    = os.path.join(SCRIPT_DIR, "outputs", "bifurcation")
TS_DIR     = os.path.join(SCRIPT_DIR, "outputs", "timeseries")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "plots", "schanz_pelster")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── File-name parsing helpers ─────────────────────────────────────────────────
def _decode_ic(s):
    """Convert 'n1.500000' → -1.5, '2.500000' → 2.5."""
    return -float(s[1:]) if s.startswith("n") else float(s)

def parse_filename(path):
    """
    Parse filename of the form:
      tau_25_k_0.165_ic_1.000000_twarmup_2000_tmeasure_5000_dt_0.05.tsv
    Returns dict or None on failure.
    """
    name = os.path.basename(path).removesuffix(".tsv")
    try:
        tau_str = name.split("tau_")[1].split("_k_")[0]
        k_str   = name.split("_k_")[1].split("_ic_")[0]
        ic_str  = name.split("_ic_")[1].split("_twarmup_")[0]
        tw_str  = name.split("_twarmup_")[1].split("_tmeasure_")[0]
        tm_str  = name.split("_tmeasure_")[1].split("_dt_")[0]
        rdt_str = name.split("_dt_")[1]
        return dict(
            tau = float(tau_str),
            k   = float(k_str),
            ic  = _decode_ic(ic_str),
            tw  = float(tw_str),
            tm  = float(tm_str),
            rdt = float(rdt_str),
        )
    except (IndexError, ValueError):
        return None

# ── Data loaders ──────────────────────────────────────────────────────────────
def load_bif_file(path, event_types=None):
    """
    Returns dict of {event_type: array_of_values}.
    Values for LOCAL_MAX/MIN are θ; for ZERO_UP/DOWN are dθ/dt.
    """
    result = {"LOCAL_MAX": [], "LOCAL_MIN": [], "ZERO_UP": [], "ZERO_DOWN": []}
    try:
        with open(path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                etype = parts[0].strip()
                val   = float(parts[2].strip())
                if etype in result:
                    result[etype].append(val)
    except Exception:
        pass
    return {k: np.array(v) for k, v in result.items()}

def load_ts_file(path):
    """Returns (t, theta) arrays."""
    try:
        data = np.loadtxt(path, skiprows=1)
        if data.ndim == 1:
            return None, None
        return data[:, 0], data[:, 1]
    except Exception:
        return None, None

# ── Build bifurcation data catalogue ─────────────────────────────────────────
def load_bif_catalogue(bif_dir, tw_filter=None, tm_filter=None):
    """
    Returns list of dicts:
        { tau, k, tauk, ic, rdt, events }
    where events = load_bif_file(...)
    """
    tw_part = int(tw_filter) if tw_filter is not None else "*"
    tm_part = int(tm_filter) if tm_filter is not None else "*"
    pattern = f"tau_{TAU}_k_*_ic_*_twarmup_{tw_part}_tmeasure_{tm_part}_dt_*.tsv"
    files = glob.glob(os.path.join(bif_dir, pattern))
    rows  = []
    for fp in files:
        meta = parse_filename(fp)
        if meta is None:
            continue
        events = load_bif_file(fp)
        rows.append(dict(
            tau  = meta["tau"],
            k    = meta["k"],
            tauk = meta["tau"] * meta["k"],
            ic   = meta["ic"],
            rdt  = meta["rdt"],
            **{et: events[et] for et in events}
        ))
    return rows

# ── Build timeseries catalogue ────────────────────────────────────────────────
def load_ts_catalogue(ts_dir, tw_filter=None, tm_filter=None):
    tw_part = int(tw_filter) if tw_filter is not None else "*"
    tm_part = int(tm_filter) if tm_filter is not None else "*"
    pattern = f"tau_{TAU}_k_*_ic_*_twarmup_{tw_part}_tmeasure_{tm_part}_dt_*.tsv"
    files = glob.glob(os.path.join(ts_dir, pattern))
    rows  = []
    for fp in files:
        meta = parse_filename(fp)
        if meta is None:
            continue
        rows.append(dict(
            tau  = meta["tau"],
            k    = meta["k"],
            tauk = meta["tau"] * meta["k"],
            ic   = meta["ic"],
            rdt  = meta["rdt"],
            path = fp,
        ))
    return rows

# ── Plot helpers ──────────────────────────────────────────────────────────────
def landmark_vlines(ax, subset=None, alpha=0.25, lw=0.8, ymin=0.0, ymax=1.0, fontsize=6):
    for label, val in LANDMARK_TAUKVALS.items():
        if subset is not None and val not in subset and label not in subset:
            continue
        ax.axvline(val, color="k", lw=lw, ls="--", alpha=alpha)
        ax.text(val, ax.get_ylim()[1] * 0.97, label,
                ha="center", va="top", fontsize=fontsize,
                rotation=90, color="k", alpha=0.6)

def annotate_pd_lines(ax, fontsize=6):
    for label, val in LANDMARK_TAUKVALS.items():
        if "PD" in label or "chaos" in label or label in ("4.24",):
            ax.axvline(val, color="crimson", lw=0.8, ls=":", alpha=0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Full bifurcation diagram (COARSE sweep, LOCAL_MAX)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_bifurcation_coarse(catalogue):
    print("\n── Figure 1: Full bifurcation diagram ──")

    ic_vals = sorted(set(r["ic"] for r in catalogue), key=abs)

    fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                             gridspec_kw={"hspace": 0.05}, sharex=True)

    # Panel A: ALL local maxima
    ax = axes[0]
    for ic in ic_vals:
        c, m = IC_PALETTE.get(ic, ("gray", "."))
        rows = [r for r in catalogue if r["ic"] == ic]
        for r in rows:
            if len(r["LOCAL_MAX"]) == 0:
                continue
            ax.plot(np.full_like(r["LOCAL_MAX"], r["tauk"]),
                    r["LOCAL_MAX"], m, color=c,
                    markersize=0.6, alpha=0.5, rasterized=True)
        # Dummy for legend
        ax.plot([], [], m, color=c, markersize=4,
                label=IC_LABEL.get(ic, f"IC={ic}"))

    ax.set_ylabel("θ  (at local maxima)", fontsize=10)
    ax.set_title("Full bifurcation diagram — local maxima of θ(t)", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, markerscale=4, ncol=2)
    ax.grid(True, alpha=0.2)
    annotate_pd_lines(ax)

    # Panel B: LOCAL_MAX ∈ [0.5, 3.5]  (zoom on first attractor family)
    ax = axes[1]
    for ic in [0.5, 1.0, -1.0]:
        c, m = IC_PALETTE.get(ic, ("gray", "."))
        rows = [r for r in catalogue if r["ic"] == ic]
        for r in rows:
            vals = r["LOCAL_MAX"]
            mask = (vals > 0.4) & (vals < 3.7)
            if not np.any(mask):
                continue
            ax.plot(np.full(int(mask.sum()), r["tauk"]),
                    vals[mask], m, color=c,
                    markersize=0.8, alpha=0.5, rasterized=True)

    ax.set_ylabel("θ  (maxima, IC branch 1)", fontsize=10)
    ax.set_ylim(0.4, 3.8)
    ax.grid(True, alpha=0.2)
    annotate_pd_lines(ax)

    # Panel C: LOCAL_MAX ∈ [1, 2]  — Poincaré section requested by user
    ax = axes[2]
    for ic in ic_vals:
        c, m = IC_PALETTE.get(ic, ("gray", "."))
        rows = [r for r in catalogue if r["ic"] == ic]
        for r in rows:
            vals = r["LOCAL_MAX"]
            mask = (vals >= 1.0) & (vals <= 2.0)
            if not np.any(mask):
                continue
            ax.plot(np.full(int(mask.sum()), r["tauk"]),
                    vals[mask], m, color=c,
                    markersize=0.8, alpha=0.5, rasterized=True)

    ax.set_xlabel("τ · k", fontsize=11)
    ax.set_ylabel("θ  (maxima in [1, 2])", fontsize=10)
    ax.set_title("Poincaré section: maxima ∈ [1, 2]", fontsize=9)
    ax.grid(True, alpha=0.2)
    annotate_pd_lines(ax)

    # Shared x-axis decorations
    xlims = (1.3, 5.6)
    for a in axes:
        a.set_xlim(*xlims)

    # Landmark annotations (bottom panel only)
    yl = axes[0].get_ylim()
    for label, val in LANDMARK_TAUKVALS.items():
        for a in axes:
            a.axvline(val, color="k", lw=0.6, ls="--", alpha=0.2)

    fig.suptitle(r"DDE: dθ/dt = −k sin(θ(t−τ)),  τ = 25", fontsize=13, y=0.98)
    out = os.path.join(PLOT_DIR, "bifurcation_coarse.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Fine bifurcation diagram (period-doubling window)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_bifurcation_fine(catalogue_fine):
    print("\n── Figure 2: Fine bifurcation diagram (PD window) ──")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True,
                             gridspec_kw={"wspace": 0.06})

    xlims = (4.08, 4.30)

    # 1×2: LOCAL_MAX only, one branch per panel
    panel_specs = [
        (axes[0], "Branch 1  (IC = ±1.0)", [1.0, -1.0]),
        (axes[1], "Branch 2  (IC = ±2.0)", [2.0, -2.0]),
    ]

    pd_vals   = [4.157, 4.165, 4.1725, 4.17375, 4.175, 4.24]
    pd_labels = ["PD₁", "PD₂", "PD₃", "PD₄", "chaos\nonset", "chaos\nend"]

    for ax, title, ic_subset in panel_specs:
        for ic in ic_subset:
            c, m = IC_PALETTE.get(ic, ("gray", "."))
            rows = [r for r in catalogue_fine if r["ic"] == ic]
            for r in rows:
                vals = r["LOCAL_MAX"]
                if len(vals) == 0:
                    continue
                ax.plot(np.full_like(vals, r["tauk"]),
                        vals, m, color=c,
                        markersize=0.6, alpha=0.6, rasterized=True)
        ax.set_xlim(*xlims)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("τ · k", fontsize=10)
        ax.grid(True, alpha=0.2)

        # PD cascade vlines
        ylims = ax.get_ylim()
        for pval, plabel in zip(pd_vals, pd_labels):
            ax.axvline(pval, color="crimson", lw=0.9, ls=":", alpha=0.7)
            ax.text(pval, ylims[1] * 0.99, plabel,
                    ha="center", va="top", fontsize=6,
                    color="crimson", alpha=0.9, rotation=90)

    axes[0].set_ylabel("θ  (at local maxima)", fontsize=10)

    fig.suptitle(
        "Period-doubling cascade  (LOCAL_MAX Poincaré section)\n"
        "τ · k ∈ [4.08, 4.30]", fontsize=11)
    out = os.path.join(PLOT_DIR, "bifurcation_fine.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Coexisting limit cycles
# ═══════════════════════════════════════════════════════════════════════════════
def fig_coexisting_lc(ts_catalogue):
    print("\n── Figure 3: Coexisting limit cycles ──")

    # Four dynamical regimes from the Schanz-Pelster sequence:
    #   2.50  – single limit cycle  (between Hopf and first split)
    #   3.90  – after first split at 3.77: two branches of same LC family
    #   4.108 – two coexisting LCs  (new LC created at 4.105, before it splits)
    #   4.13  – second LC has split into 2  (after 4.11)
    target_taukvals = [2.50, 3.90, 4.108, 4.13]
    row_labels = [
        "Single limit cycle  (τ·k = 2.50)",
        "LC splits into two  (τ·k ≈ 3.90, after 3.77)",
        "Two coexisting LCs  (τ·k ≈ 4.108, just above new LC onset 4.105)",
        "Second LC splits    (τ·k ≈ 4.13, after 4.11)",
    ]
    ics_all = [1.0, -1.0, 2.0, -2.0]   # always plot all four

    fn_rows = {(r["tauk"], r["ic"]): r for r in ts_catalogue}

    fig, axes = plt.subplots(len(target_taukvals), 2,
                             figsize=(13, 4.0 * len(target_taukvals)),
                             gridspec_kw={"wspace": 0.15})

    for row_i, (tk_target, row_label) in enumerate(
            zip(target_taukvals, row_labels)):

        avail_tauks = sorted(set(r["tauk"] for r in ts_catalogue))
        if not avail_tauks:
            continue
        tk_actual = min(avail_tauks, key=lambda x: abs(x - tk_target))
        k_actual  = tk_actual / TAU

        ax_ts    = axes[row_i, 0]
        ax_phase = axes[row_i, 1]

        for ic in ics_all:
            col, _ = IC_PALETTE.get(ic, ("gray", "."))
            row = fn_rows.get((tk_actual, ic))
            if row is None:
                cands = [(r["tauk"], r) for r in ts_catalogue if r["ic"] == ic]
                if not cands:
                    continue
                _, row = min(cands, key=lambda x: abs(x[0] - tk_target))
            t, theta = load_ts_file(row["path"])
            if t is None:
                continue
            # Timeseries: last 8 base periods (window = 8 × 4τ)
            window = 8 * 4 * TAU
            mask   = t >= t[-1] - window
            ax_ts.plot(t[mask] - t[mask][0], theta[mask],
                       lw=1.2, color=col, label=f"IC = {ic:+.1f}")
            # Phase portrait: θ vs dθ/dt  (numerical derivative)
            dtheta = np.gradient(theta[mask], t[mask])
            ax_phase.plot(theta[mask], dtheta,
                          lw=0.6, color=col, alpha=0.7,
                          label=f"IC = {ic:+.1f}")

        ax_ts.set_title(f"{row_label}\nτ·k = {tk_actual:.4f}  (k = {k_actual:.5f})",
                        fontsize=8)
        ax_ts.set_xlabel("time  (last 8 periods)")
        ax_ts.set_ylabel("θ(t)")
        ax_ts.legend(fontsize=8)
        ax_ts.grid(True, alpha=0.3)

        ax_phase.set_xlabel(r"$\theta$")
        ax_phase.set_ylabel(r"$d\theta/dt$")
        ax_phase.legend(fontsize=8)
        ax_phase.grid(True, alpha=0.3)

    fig.suptitle(
        r"DDE: d$\theta$/dt = $-k\sin(\theta(t-\tau))$  ·  Four dynamical regimes",
        fontsize=11)
    out = os.path.join(PLOT_DIR, "coexisting_lc.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Period-doubling timeseries
# ═══════════════════════════════════════════════════════════════════════════════
def fig_period_doubling_ts(ts_catalogue):
    print("\n── Figure 4: Period-doubling timeseries ──")

    # Show key τ·k through the PD cascade of the second LC branch (starts at 4.11)
    target_tauks = [4.08, 4.11, 4.13, 4.15, 4.157, 4.165, 4.1725, 4.178]
    ic_show = 2.0   # branch 2 shows PD first (starting at τ·k ≈ 4.11)

    avail_by_ic = {r["tauk"]: r for r in ts_catalogue if r["ic"] == ic_show}
    if not avail_by_ic:
        print("  No timeseries with IC=2.0 found; skipping.")
        return

    selected = []
    for tk in target_tauks:
        closest_tk = min(avail_by_ic.keys(), key=lambda x: abs(x - tk))
        if abs(closest_tk - tk) < 0.05:
            selected.append((tk, closest_tk, avail_by_ic[closest_tk]))

    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, (tk_target, tk_actual, row) in enumerate(selected):
        t, theta = load_ts_file(row["path"])
        ax = axes_flat[i]
        if t is None:
            ax.set_visible(False)
            continue
        # Show 12 periods = 12 × 4τ
        window = 12 * 4 * TAU
        mask   = t >= t[-1] - window
        ax.plot(t[mask] - t[mask][0], theta[mask], lw=0.9, color="#2166AC")
        ax.set_title(f"τ·k = {tk_actual:.4f}", fontsize=9)
        ax.set_xlabel("time  (last 12 periods)")
        ax.set_ylabel("θ(t)")
        ax.grid(True, alpha=0.3)

        # Mark expected period multiples
        T0 = 4 * TAU   # base period ≈ 4τ = 100
        for n_T in range(0, int(window / T0) + 1):
            ax.axvline(n_T * T0, color="gray", lw=0.5, ls=":", alpha=0.4)

    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Period-doubling cascade  (IC = +2.0,  branch 2)\n"
        "Vertical dotted lines mark multiples of the base period T₀ = 4τ = 100",
        fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "period_doubling_ts.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Power spectra (Welch PSD)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_power_spectra(ts_catalogue):
    print("\n── Figure 5: Power spectra ──")

    # Branch 2 shows PD starting at τ·k ≈ 4.11, earlier than branch 1
    target_tauks = [4.08, 4.11, 4.13, 4.15, 4.157, 4.165, 4.1725, 4.178]
    ic_show = 2.0

    avail = {r["tauk"]: r for r in ts_catalogue if r["ic"] == ic_show}
    if not avail:
        print("  No timeseries with IC=2.0 found; skipping.")
        return

    selected = []
    for tk in target_tauks:
        closest_tk = min(avail.keys(), key=lambda x: abs(x - tk))
        if abs(closest_tk - tk) < 0.05:
            selected.append((tk, closest_tk, avail[closest_tk]))

    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    f0 = 1.0 / (4 * TAU)   # fundamental frequency (period = 4τ)

    for i, (tk_target, tk_actual, row) in enumerate(selected):
        t, theta = load_ts_file(row["path"])
        ax = axes_flat[i]
        if t is None or len(theta) < 512:
            ax.set_visible(False)
            continue

        dt_ts   = t[1] - t[0]
        fs      = 1.0 / dt_ts
        # Welch PSD — large segments for sharp spectral peaks
        nperseg  = min(len(theta) // 2, 32768)
        noverlap = nperseg // 2
        freqs, psd = welch(theta, fs=fs, nperseg=nperseg, noverlap=noverlap,
                           window="hann", detrend="constant")
        ax.semilogy(freqs / f0, psd, lw=0.8, color="#2166AC")

        # Mark expected subharmonics: f0, f0/2, f0/4
        for sub, lbl in [(1, "f₀"), (0.5, "f₀/2"), (0.25, "f₀/4"), (0.125, "f₀/8")]:
            ax.axvline(sub, color="crimson", lw=0.8, ls="--", alpha=0.7)
            ax.text(sub, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 1,
                    lbl, ha="center", va="top", fontsize=6,
                    color="crimson", rotation=90, alpha=0.8)

        ax.set_title(f"τ·k = {tk_actual:.4f}", fontsize=9)
        ax.set_xlabel("Frequency / f₀", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.set_xlim(0, 3)
        ax.grid(True, alpha=0.2, which="both")

    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Power spectral density (Welch)  ·  IC = +2.0 (branch 2)  ·  f₀ = 1/(4τ)\n"
        "Subharmonic peaks at f₀/2, f₀/4 signal period doubling",
        fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "power_spectra.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Return maps / stroboscopic Poincaré sections
# ═══════════════════════════════════════════════════════════════════════════════
def fig_return_maps(ts_catalogue):
    print("\n── Figure 6: Return maps ──")

    # Cover fixed point, LC, onset of coexistence, PD1, PD2, chaos, post-chaos
    target_tauks = [1.40, 2.50, 3.77, 4.00, 4.10, 4.157, 4.165, 4.178, 4.24]
    ic_pairs     = [(0.5, 1.0), (0.5, 1.0), (1.0, 2.0), (1.0, 2.0),
                    (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]

    avail_tauks = sorted(set(r["tauk"] for r in ts_catalogue))

    nplots = len(target_tauks)
    ncols  = 3
    nrows  = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat  = axes.flatten()

    strobe_dt = 4 * TAU

    for pi, (tk_target, ic_pair) in enumerate(zip(target_tauks, ic_pairs)):
        if not avail_tauks:
            continue
        tk_actual = min(avail_tauks, key=lambda x: abs(x - tk_target))
        k_actual  = tk_actual / TAU
        ax        = axes_flat[pi]

        plotted = False
        for ic in ic_pair:
            cands = [r for r in ts_catalogue if r["ic"] == ic and abs(r["tauk"] - tk_target) < 0.05]
            if not cands:
                continue
            row = min(cands, key=lambda r: abs(r["tauk"] - tk_target))
            t, theta = load_ts_file(row["path"])
            if t is None:
                continue
            n_max   = int((t[-1] - t[0]) / strobe_dt)
            targets = t[0] + np.arange(1, n_max + 1) * strobe_dt
            idx     = np.searchsorted(t, targets)
            idx     = np.clip(idx, 0, len(t) - 1)
            th_n    = theta[idx]
            c, m    = IC_PALETTE.get(ic, ("gray", "."))
            ax.plot(th_n[:-1], th_n[1:], m, color=c,
                    markersize=1.5, alpha=0.6, label=f"IC={ic:+.1f}",
                    rasterized=True)
            plotted = True

        if plotted:
            lo, hi = ax.get_xlim()
            diag = np.linspace(lo, hi, 100)
            ax.plot(diag, diag, "k--", lw=0.8, label="y = x", alpha=0.5)

        ax.set_title(f"τ·k = {tk_actual:.4f}", fontsize=9)
        ax.set_xlabel(r"$\theta_n$", fontsize=9)
        ax.set_ylabel(r"$\theta_{n+1}$", fontsize=9)
        ax.legend(fontsize=7, markerscale=3)
        ax.grid(True, alpha=0.3)

    for j in range(nplots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Stroboscopic return maps  (Δt = 4τ = {int(4*TAU)})\n"
        "Fixed pt → LC → two coexisting LCs → PD cascade → chaos",
        fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "return_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Mean period (to verify Hopf and period doubling from COARSE data)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_mean_period(catalogue):
    """
    Estimate the mean period from consecutive ZERO_UP crossing times and
    plot <T> vs τ·k.  Period doubling shows up as sudden doublings in <T>.
    """
    print("\n── Figure 7: Mean period from zero-crossings ──")

    all_files = glob.glob(os.path.join(BIF_DIR, "*.tsv"))
    T0 = 4 * TAU   # expected base period

    fig, ax = plt.subplots(figsize=(12, 5))

    any_data = False
    for ic_show in [2.0, -2.0]:
        col, _ = IC_PALETTE.get(ic_show, ("gray", "."))
        ic_rows = sorted([r for r in catalogue if r["ic"] == ic_show],
                         key=lambda r: r["tauk"])
        if not ic_rows:
            continue

        meta_map = {}
        for fp in all_files:
            m = parse_filename(fp)
            if m and abs(m["ic"] - ic_show) < 0.01:
                meta_map[(m["k"], m["ic"])] = (m["tm"], fp)

        tk_arr = []
        T_mean = []
        T_std  = []

        for r in ic_rows:
            key = (r["k"], r["ic"])
            if key not in meta_map:
                continue
            _t_measure, fp = meta_map[key]

            times_xu = []
            try:
                with open(fp) as f:
                    next(f)
                    for line in f:
                        parts = line.split("\t")
                        if len(parts) < 3:
                            continue
                        if parts[0].strip() == "ZERO_UP":
                            times_xu.append(float(parts[1].strip()))
            except Exception:
                continue

            if len(times_xu) < 3:
                continue

            t_arr_ic  = np.array(times_xu)
            intervals = np.diff(t_arr_ic)
            med       = np.median(intervals)
            good      = intervals[(intervals > 0.1 * med) & (intervals < 5 * T0)]
            if len(good) < 2:
                continue
            tk_arr.append(r["tauk"])
            T_mean.append(np.mean(good))
            T_std.append(np.std(good))

        if not tk_arr:
            continue

        any_data = True
        tk_arr = np.array(tk_arr)
        T_arr  = np.array(T_mean) / T0
        T_err  = np.array(T_std)  / T0
        ax.errorbar(tk_arr, T_arr, yerr=T_err, fmt=".", markersize=4,
                    color=col, elinewidth=0.5, capsize=2,
                    label=IC_LABEL.get(ic_show, f"IC={ic_show:+.1f}"))

    if not any_data:
        print("  No crossing-time data; skipping.")
        plt.close(fig)
        return

    for n in [1, 2, 4, 8]:
        ax.axhline(n, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.text(1.35, n + 0.02, f"T = {n}·T₀", fontsize=7, color="gray")

    for lbl, val in LANDMARK_TAUKVALS.items():
        ax.axvline(val, color="k", lw=0.6, ls=":", alpha=0.25)

    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("τ · k", fontsize=11)
    ax.set_ylabel("<T> / T₀  (normalised mean period)", fontsize=10)
    ax.set_title("Mean period vs τ·k  (IC = ±2.0,  branch 2)\n"
                 "Doublings at PD bifurcations; chaotic window → irregular T",
                 fontsize=10)
    ax.set_xlim(1.3, 5.6)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "mean_period.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Output figures → {PLOT_DIR}")
    print(f"Reading bifurcation data from {BIF_DIR}")
    print(f"Reading timeseries data from   {TS_DIR}")

    # Load data catalogues
    # COARSE: twarmup=8000, tmeasure=20000
    cat_coarse = load_bif_catalogue(BIF_DIR, tw_filter=8000, tm_filter=20000)
    print(f"\nCoarse catalogue: {len(cat_coarse)} entries")

    # FINE: twarmup=20000, tmeasure=100000
    cat_fine = load_bif_catalogue(BIF_DIR, tw_filter=20000, tm_filter=100000)
    print(f"Fine catalogue:   {len(cat_fine)} entries")

    # Timeseries: twarmup=10000, tmeasure=500000
    cat_ts = load_ts_catalogue(TS_DIR, tw_filter=10000, tm_filter=500000)
    print(f"Timeseries catalogue: {len(cat_ts)} entries")

    if len(cat_coarse) == 0 and len(cat_fine) == 0:
        print("\nNo simulation data found.  Run  bash runBifurcation.sh  first.")
        return

    # Produce figures
    if cat_coarse:
        fig_bifurcation_coarse(cat_coarse)
        fig_mean_period(cat_coarse)
    else:
        print("  [skip] No coarse data → bifurcation_coarse, mean_period")

    if cat_fine:
        fig_bifurcation_fine(cat_fine)
    else:
        print("  [skip] No fine data → bifurcation_fine")

    if cat_ts:
        fig_coexisting_lc(cat_ts)
        fig_period_doubling_ts(cat_ts)
        fig_power_spectra(cat_ts)
        fig_return_maps(cat_ts)
    else:
        print("  [skip] No timeseries data → coexisting_lc, period_doubling_ts, "
              "power_spectra, return_maps")

    print("\nDone.")


if __name__ == "__main__":
    main()
