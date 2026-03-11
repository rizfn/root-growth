import numpy as np
import matplotlib.pyplot as plt
import os, glob, re, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FUNC_NAME  = "lnp1"
FUNC_LABEL = r"$f(x) = \mathrm{sign}(x)\,\ln(1+|x|)$"
TAU       = 1.0
RECORD_DT = 0.1


def load_timeseries(path):
    try:
        data  = np.loadtxt(path, skiprows=1, usecols=(0, 1))
        mask  = np.isfinite(data[:, 1])
        return data[mask, 0], data[mask, 1]
    except Exception:
        return None, None


def main():
    files = sorted(glob.glob(f"{SCRIPT_DIR}/outputs/{FUNC_NAME}/k_*.tsv"))
    if not files:
        print(f"No output files found in outputs/{FUNC_NAME}/. Run run_{FUNC_NAME}.sh first.")
        return

    ks, amps, paths = [], [], []
    for f in files:
        m = re.search(r"/k_([^_]+)_ic_", f)
        if m:
            ks.append(float(m.group(1)))
            paths.append(f)
            _, theta = load_timeseries(f)
            amps.append(np.max(np.abs(theta[len(theta) // 2:])) if theta is not None and len(theta) else np.nan)

    ks    = np.array(ks)
    amps  = np.array(amps)
    idx   = np.argsort(ks)
    ks, amps, paths = ks[idx], amps[idx], [paths[i] for i in idx]

    out_dir = f"{SCRIPT_DIR}/plots/{FUNC_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    # ── Amplitude vs k ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, amps, "o-", ms=3, lw=1)
    ax.axvline(math.pi / 2, ls="--", c="gray", alpha=0.6,
               label=rf"$k_{{\rm Hopf}}={math.pi/2:.3f}$ (linear)")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"max$|\theta|$ (steady state)")
    ax.set_title(rf"d$\theta$/dt $= -k\,f(\theta(t-\tau))$,  {FUNC_LABEL},  $\tau={TAU}$")
    ax.legend()
    plt.tight_layout()
    out_amp = f"{out_dir}/amplitude_{FUNC_NAME}.png"
    plt.savefig(out_amp, dpi=150)
    print(f"Saved: {out_amp}")

    # ── Timeseries: 4 sampled k values ──
    n          = len(ks)
    sample_idx = [n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5]
    tail_pts   = int(round(20 * TAU / RECORD_DT))

    fig2, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)
    for ax, si in zip(axes, sample_idx):
        t, theta = load_timeseries(paths[si])
        if theta is None:
            ax.set_visible(False)
            continue
        t_tail     = t[-tail_pts:]
        theta_tail = theta[-tail_pts:]
        ax.plot(t_tail - t_tail[0], theta_tail, lw=0.8)
        ax.set_title(rf"$k={ks[si]:.2f}$")
        ax.set_xlabel(r"$t - t_0$")
        ax.set_ylabel(r"$\theta$")
    fig2.suptitle(rf"Timeseries (last $20\tau$),  {FUNC_LABEL}")
    plt.tight_layout()
    out_ts = f"{out_dir}/timeseries_{FUNC_NAME}.png"
    plt.savefig(out_ts, dpi=150)
    print(f"Saved: {out_ts}")
    plt.show()


if __name__ == "__main__":
    main()
