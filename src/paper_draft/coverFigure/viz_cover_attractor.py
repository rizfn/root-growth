"""Cover figure: the chaotic attractor of

    dtheta/dt = -k * sin(theta(t - tau)),   tau = 1,

on the long-period (LC2) branch, tau*k = 4.235, where the largest Lyapunov
exponent is ~0.21 / tau.

The attractor is shown in the delay embedding

    (theta(t), theta(t - lag_1), theta(t - lag_2)),

rotated and projected orthographically.  Each pixel accumulates how often the
trajectory passes through it (brightness, log-scaled) and the mean depth of
those passes (colour: far strands cool and dim, near strands warm and bright).
A soft bloom is added on top.  No axes: this is an image, not a plot.

Input comes from runCover.sh (raw float64 theta, one value every record_dt).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "plots")

PALETTES = {
    # far -> near
    "ember": ["#140d3c", "#3b1a73", "#8a2a8a", "#d4446a", "#f08a3c", "#ffd27a"],
    "root": ["#0b2238", "#0f4d5c", "#1f8a7a", "#6fbf73", "#d8d76a", "#fff3b0"],
    "ice": ["#10173a", "#1f3f7a", "#2f7fb0", "#6cc3d5", "#c9f0ee", "#ffffff"],
}
BACKGROUND = {"ember": "#07050f", "root": "#040b0e", "ice": "#05070f"}


def rotation(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(el), -np.sin(el)], [0.0, np.sin(el), np.cos(el)]])
    return rx @ rz


def accumulate(theta, lags, rot, width, height, box, substeps, split=None, n_chunk=5_000_000):
    """Counts and summed nearness per pixel, for the projected delay embedding.

    The attractor is fitted into the frame minus the free fractions
    box = (left, right, top, bottom); a large `top` leaves room for the
    masthead.  With `split` the points are divided into a back layer
    (nearness < split) and a front layer, so that other things can be
    sandwiched between them; otherwise there is a single layer.
    Nearness runs from 0 (farthest strands) to 1 (nearest).
    """
    left, right, top, bottom = box
    lag_max = max(lags)
    n = len(theta) - lag_max
    centre = np.array([theta.mean()] * 3)

    # First pass on a subsample: the screen-space extent, so the attractor
    # fills its box.
    sub = np.arange(lag_max, len(theta), 97)
    p = np.stack([theta[sub - l] for l in lags], axis=1) - centre
    q = p @ rot.T
    sx, sy, sd = q[:, 0], q[:, 2], q[:, 1]
    xmid, ymid = 0.5 * (sx.max() + sx.min()), 0.5 * (sy.max() + sy.min())
    box_w, box_h = (1.0 - left - right) * width, (1.0 - top - bottom) * height
    scale = min(box_w / (sx.max() - sx.min()), box_h / (sy.max() - sy.min()))  # pixels per unit
    x_centre = (0.5 + 0.5 * (left - right)) * width
    y_centre = (0.5 + 0.5 * (bottom - top)) * height
    dlo, dhi = np.percentile(sd, 0.5), np.percentile(sd, 99.5)

    n_layers = 1 if split is None else 2
    counts = [np.zeros(width * height) for _ in range(n_layers)]
    nearness = [np.zeros(width * height) for _ in range(n_layers)]
    for start in range(lag_max, lag_max + n - 1, n_chunk):
        stop = min(start + n_chunk + 1, lag_max + n)
        idx = np.arange(start, stop)
        p = np.stack([theta[idx - l] for l in lags], axis=1) - centre
        q_all = p @ rot.T
        # Consecutive samples are several pixels apart, which leaves the strands
        # stippled; fill in `substeps` evenly spaced points (in time) between them.
        for f in np.arange(substeps) / substeps:
            q = (1.0 - f) * q_all[:-1] + f * q_all[1:]
            px = ((q[:, 0] - xmid) * scale + x_centre).astype(np.int64)
            py = ((q[:, 2] - ymid) * scale + y_centre).astype(np.int64)
            ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            flat = (height - 1 - py[ok]) * width + px[ok]
            # the depth axis points away from the viewer
            near = 1.0 - np.clip((q[ok, 1] - dlo) / (dhi - dlo), 0.0, 1.0)
            layer = np.zeros(len(flat), dtype=int) if split is None else (near >= split).astype(int)
            for i in range(n_layers):
                sel = layer == i
                counts[i] += np.bincount(flat[sel], minlength=width * height)
                nearness[i] += np.bincount(flat[sel], weights=near[sel], minlength=width * height)
    return [c.reshape(height, width) for c in counts], [d.reshape(height, width) for d in nearness]


def downsample(a, s):
    if s == 1:
        return a
    h, w = a.shape
    return a.reshape(h // s, s, w // s, s).sum(axis=(1, 3))


def emissive(counts, nearness, palette, gamma, exposure, bloom, supersample):
    """Light emitted by each layer (RGB on black), normalised jointly."""
    counts = [downsample(c, supersample) for c in counts]
    nearness = [downsample(d, supersample) for d in nearness]
    total = sum(counts)
    nz = total[total > 0]
    ref = np.percentile(nz, 50)
    top = np.log1p(np.percentile(nz, 99.8) / ref)
    cmap = mcolors.LinearSegmentedColormap.from_list("cover", PALETTES[palette])

    layers = []
    for c, d in zip(counts, nearness):
        near = np.divide(d, c, out=np.zeros_like(d), where=c > 0)
        # Fill empty pixels' depth from their neighbourhood, so the bloom does
        # not pick up a spurious far colour at the edges of the ribbon.
        smooth_c, smooth_d = gaussian_filter(c, 2.0), gaussian_filter(d, 2.0)
        near = np.where(c > 0, near, np.divide(smooth_d, smooth_c, out=np.zeros_like(smooth_d), where=smooth_c > 0))

        lum = np.clip(np.log1p(c / ref) / top, 0.0, 1.0) ** gamma
        lum *= 0.4 + 0.6 * near  # far strands recede
        lum = (1.0 - np.exp(-exposure * lum)) / (1.0 - np.exp(-exposure))  # soft shoulder
        rgb = cmap(0.15 + 0.85 * near)[..., :3] * lum[..., None]
        layers.append(add_glow(rgb, bloom))
    return layers


def add_glow(rgb, bloom, scale=0.004):
    sigma = scale * max(rgb.shape[:2])
    glow = sum(w * gaussian_filter(rgb, sigma=(f * sigma, f * sigma, 0)) for f, w in ((1.0, 0.6), (4.0, 0.4)))
    return screen(rgb, bloom * glow)


def screen(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


def background(height, width, palette, lift=1.2, centre_y=0.55):
    """The palette's ground colour, lifted slightly towards the centre."""
    yy, xx = np.mgrid[0:height, 0:width]
    r2 = ((xx - 0.5 * width) / (0.5 * width)) ** 2 + ((yy - centre_y * height) / (0.5 * height)) ** 2
    bg = np.array(mcolors.to_rgb(BACKGROUND[palette]))
    return bg[None, None, :] * (1.0 + lift * np.exp(-r2 / 0.6))[..., None]


def save_image(img, base, width_in, height_in, dpi):
    plt.imsave(base + ".png", img, dpi=dpi)
    if dpi < 150:
        return
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    fig.savefig(base + ".pdf", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {base}.png / .pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw float64 theta from attractorRun")
    ap.add_argument("--tauk", type=float, default=4.235)
    ap.add_argument("--record-dt", type=float, default=0.002)
    ap.add_argument("--lags", type=float, nargs=3, default=[0.0, 1.0, 2.0], help="delays in units of tau")
    ap.add_argument("--az", type=float, default=60.0)
    ap.add_argument("--el", type=float, default=35.0)
    ap.add_argument("--palette", default="ember", choices=sorted(PALETTES))
    ap.add_argument("--width-in", type=float, default=8.5)
    ap.add_argument("--height-in", type=float, default=11.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--top", type=float, default=0.20, help="free fraction at the top, for the masthead")
    ap.add_argument("--bottom", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--exposure", type=float, default=3.0)
    ap.add_argument("--bloom", type=float, default=0.55)
    ap.add_argument("--substeps", type=int, default=4, help="interpolated points per recorded interval")
    ap.add_argument("--max-points", type=int, default=0, help="use only the first N samples (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    theta = np.fromfile(args.input, dtype=np.float64)
    if args.max_points:
        theta = theta[: args.max_points]
    lags = [int(round(l / args.record_dt)) for l in args.lags]

    s = args.supersample
    width = int(round(args.width_in * args.dpi)) * s
    height = int(round(args.height_in * args.dpi)) * s
    box = (args.margin, args.margin, args.top, args.bottom)
    counts, nearness = accumulate(theta, lags, rotation(args.az, args.el), width, height, box, args.substeps)
    (layer,) = emissive(counts, nearness, args.palette, args.gamma, args.exposure, args.bloom, s)
    img = np.clip(screen(layer, background(*layer.shape[:2], args.palette)), 0.0, 1.0)

    os.makedirs(args.out_dir, exist_ok=True)
    lag_str = "_".join(f"{l:g}" for l in args.lags)
    base = os.path.join(
        args.out_dir,
        f"cover_attractor_tauk_{args.tauk:g}_lags_{lag_str}_az_{args.az:g}_el_{args.el:g}"
        f"_{args.palette}_{args.width_in:g}x{args.height_in:g}in_{args.dpi}dpi{args.tag}",
    )
    save_image(img, base, args.width_in, args.height_in, args.dpi)


if __name__ == "__main__":
    main()
