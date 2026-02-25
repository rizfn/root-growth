"""
Visualize outputs from noiseHeunPeriodicIC.
Globs all .tsv files under outputs/SDDETimeseries/periodic_ic/ and makes
one plot per file: left panel (2/3 width) shows IC + first 5τ, right panel (1/3) shows last 4τ.
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "outputs", "SDDETimeseries", "periodic_ic")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "plots", "periodic_ic")

# Regex patterns to extract parameters from paths
RE_FOLDER = re.compile(
    r"tau_(?P<tau>[\d.]+)_k_(?P<k>[\d.]+)_dt_(?P<dt>[\d.]+)_tmax_(?P<tmax>[\d.]+)"
)
RE_FILE = re.compile(
    r"amp_(?P<amp>[\d.]+)_period_(?P<period>[\d.]+)_eta_(?P<eta>[\d.eE+-]+)_simNo_(?P<sim_no>\d+)\.tsv"
)


def parse_path(tsv_path):
    """Return dict of parameters parsed from folder + filename."""
    folder = os.path.basename(os.path.dirname(tsv_path))
    fname  = os.path.basename(tsv_path)
    mf = RE_FOLDER.search(folder)
    mn = RE_FILE.search(fname)
    if not mf or not mn:
        return None
    return {
        "tau":    float(mf.group("tau")),
        "k":      float(mf.group("k")),
        "dt":     float(mf.group("dt")),
        "tmax":   float(mf.group("tmax")),
        "amp":    float(mn.group("amp")),
        "period": float(mn.group("period")),
        "eta":    float(mn.group("eta")),
        "sim_no": int(mn.group("sim_no")),
    }



def plot_individual(tsv_path, params, out_dir):
    """Two-panel plot: left (2/3) = IC + first 5τ, right (1/3) = last 4τ."""
    data  = np.loadtxt(tsv_path, skiprows=1)
    t     = data[:, 0]
    theta = data[:, 1]
    tmax  = params["tmax"]
    tau   = params["tau"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 4), gridspec_kw={"width_ratios": [2, 1]}
    )

    # ── Left: IC (red) + early dynamics up to 5τ ──────────────────
    mask_l = (t >= -tau) & (t <= 5 * tau)
    t_l, th_l = t[mask_l], theta[mask_l]
    ax_left.plot(t_l[t_l <= 0], th_l[t_l <= 0], color="C3", linewidth=1.0,
                 alpha=0.9, label="IC (sinusoidal)")
    ax_left.plot(t_l[t_l >= 0], th_l[t_l >= 0], color="C0", linewidth=0.7,
                 alpha=0.85, label="dynamics")
    ax_left.axvline(0, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_left.set_xlabel("Time")
    ax_left.set_ylabel("θ (rad)")
    ax_left.set_title(f"IC + early dynamics  [−τ, 5τ]", fontsize=9)
    ax_left.legend(fontsize=7)
    ax_left.grid(True, alpha=0.3)

    # ── Right: steady state — last 4τ ──────────────────────────────
    mask_r = t >= tmax - 4 * tau
    t_r, th_r = t[mask_r], theta[mask_r]
    ax_right.plot(t_r, th_r, color="C0", linewidth=0.7, alpha=0.85)
    ax_right.set_xlabel("Time")
    ax_right.set_ylabel("θ (rad)")
    ax_right.set_title(f"Steady state  [T−4τ, T]", fontsize=9)
    ax_right.grid(True, alpha=0.3)

    p = params
    fig.suptitle(
        f"τ={p['tau']}  k={p['k']}  A={p['amp']}  T_ic={p['period']}  η={p['eta']}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(out_dir, exist_ok=True)
    fname = (
        f"tau_{p['tau']}_k_{p['k']}_amp_{p['amp']}_period_{p['period']}"
        f"_eta_{p['eta']}.png"
    )
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    tsv_files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*.tsv"), recursive=True))
    if not tsv_files:
        print(f"No .tsv files found under {BASE_DIR}")
        return
    print(f"Found {len(tsv_files)} tsv file(s).")

    entries = []
    for tsv in tsv_files:
        p = parse_path(tsv)
        if p is None:
            print(f"  [skip] Could not parse: {tsv}")
            continue
        p["path"] = tsv
        entries.append(p)

    # Individual plots
    for e in entries:
        out_sub = os.path.join(PLOT_DIR, "individual", f"tau_{e['tau']}_k_{e['k']}")
        out = plot_individual(e["path"], e, out_sub)
        print(f"  saved: {os.path.relpath(out, SCRIPT_DIR)}")

    print("Done.")


if __name__ == "__main__":
    main()
