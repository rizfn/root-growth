"""Cover figure options: the chaotic attractor together with real Arabidopsis
seedlings cut out of the plate photographs (extract_roots.py).

Everything is drawn as light on a dark ground and combined with a screen
blend, so the photographs keep their texture (root hairs, translucency) but
take on the colours of the attractor's palette.  Options:

  curtain   an ensemble of seedlings hanging from the top in three depth tiers
            (far ones small, dim and out of focus), the attractor in front;
  hero      one large seedling down the left of the cover, with the attractor
            ribbon wrapped around it (the near half of the ribbon passes in
            front of the root, the far half behind), over a faint ensemble;
  hero2     the same with a larger, branching seedling;
  plate     a row of seedlings as on the agar plate, in front of the ribbon;
  descent   a row of seedlings in the upper half whose roots fade out as they
            grow down into the attractor below.

All covers are square.

Inputs: outputs/root_crops.npz (extract_roots.py) and the trajectory from
runCover.sh.
"""

import argparse
import os

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from viz_cover_attractor import accumulate, add_glow, background, emissive, rotation, save_image, screen

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "plots", "with_roots")
CROPS = os.path.join(SCRIPT_DIR, "outputs", "root_crops.npz")

TAUK = 4.235
RECORD_DT = 0.002
TRAJECTORY = os.path.join(SCRIPT_DIR, "outputs", f"tauk_{TAUK}_theta0_2_tmeasure_100000_rdt_{RECORD_DT}.bin")
LAGS = (0.0, 1.0, 2.0)
AZ, EL = 60.0, 35.0
PALETTE = "ember"

# Colour ramps for the photographs, dark -> bright.  Far tiers stay in the cool
# end of the ember palette so they read as background; the hero root is warm.
ROOT_RAMPS = {
    "far": ["#140d3c", "#2c1660", "#4a1d6e", "#6a2380"],
    "mid": ["#1b1446", "#3b1a73", "#8a2a8a", "#b8367a"],
    "near": ["#2a1250", "#8a2a8a", "#d4446a", "#f08a3c"],
    "hero": ["#3b1a73", "#b8367a", "#f08a3c", "#ffd27a", "#fff1d0"],
}

HERO = 8           # small cotyledons and a clearly waving primary root
HERO_ALT = 109     # large cotyledons, branching
# Seedlings picked by eye for the prominent roles (filled cotyledons, a clear
# primary root); the segmentation of plate 'Rome-1' outlines the cotyledons
# instead of filling them, so those crops are left out altogether.
CURATED = [8, 9, 15, 18, 37, 100, 103, 106, 109, 111, 133, 145, 196, 211, 212, 214, 222, 224, 248]
EXCLUDE = set(range(203, 211))


# ── seedlings ────────────────────────────────────────────────────────────────

def load_seedlings():
    d = np.load(CROPS)
    out = []
    for i in range(len(d["source"])):
        gray, mask = d[f"gray_{i}"].astype(float), d[f"mask_{i}"].astype(float) / 255.0
        h, w = gray.shape
        # Whole seedlings only; very narrow components are plate-edge artefacts.
        if h < 1000 or w < 40 or w > 0.45 * h or i in EXCLUDE:
            continue
        inside = gray[mask > 0.5]
        lo, hi = np.percentile(inside, 2), np.percentile(inside, 99.5)
        value = np.clip((gray - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        out.append({"id": i, "value": value, "mask": mask})
    return out


def render_seedling(s, height_px, ramp, brightness, blur_px, flip, gamma=0.8):
    """Emission (RGB) and soft alpha of one seedling scaled to height_px."""
    h, w = s["value"].shape
    scale = height_px / h
    size = (max(2, int(round(w * scale))), max(2, int(round(h * scale))))
    value = np.asarray(Image.fromarray((s["value"] * 255).astype(np.uint8)).resize(size, Image.LANCZOS)) / 255.0
    alpha = np.asarray(Image.fromarray((s["mask"] * 255).astype(np.uint8)).resize(size, Image.LANCZOS)) / 255.0
    alpha = gaussian_filter(alpha, 0.6 + blur_px)
    value = gaussian_filter(value, blur_px) if blur_px > 0 else value
    if flip:
        value, alpha = value[:, ::-1], alpha[:, ::-1]
    cmap = mcolors.LinearSegmentedColormap.from_list("root", ramp)
    v = np.clip(value, 0.0, 1.0) ** gamma
    rgb = cmap(v)[..., :3] * (brightness * alpha * (0.35 + 0.65 * v))[..., None]
    return rgb, alpha


def paste(canvas, layer, x_centre, y_top):
    """Screen-blend `layer` into `canvas` with its top-centre at (x_centre, y_top)."""
    H, W = canvas.shape[:2]
    h, w = layer.shape[:2]
    x0, y0 = int(round(x_centre - w / 2)), int(round(y_top))
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = layer[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    if canvas.ndim == 3:
        canvas[cy0:cy1, cx0:cx1] = screen(canvas[cy0:cy1, cx0:cx1], sub)
    else:
        canvas[cy0:cy1, cx0:cx1] = np.maximum(canvas[cy0:cy1, cx0:cx1], sub)


def curtain(H, W, seedlings, rng, tiers, fade=None):
    """Ensemble of seedlings hanging from the top.

    tiers: list of dicts with keys n, height (fraction of H, range), ramp,
    brightness, blur (pixels at 300 dpi, scaled with H), top (fraction range).
    fade: optional (start, end) fractions of H over which the roots fade out.
    """
    layer = np.zeros((H, W, 3))
    px_scale = H / 3300.0
    curated = [s for s in seedlings if s["id"] in CURATED]
    for tier in tiers:
        n = tier["n"]
        pool = curated if tier.get("curated") else seedlings
        # evenly spread with jitter, so the curtain has no gaps or clumps
        xs = (np.arange(n) + 0.5 + rng.uniform(-0.35, 0.35, n)) / n * W * 1.1 - 0.05 * W
        for x in rng.permutation(xs):
            s = pool[rng.integers(len(pool))]
            hpx = rng.uniform(*tier["height"]) * H
            rgb, _ = render_seedling(s, hpx, ROOT_RAMPS[tier["ramp"]], tier["brightness"] * rng.uniform(0.75, 1.0),
                                     tier["blur"] * px_scale, rng.random() < 0.5)
            paste(layer, rgb, x, rng.uniform(*tier["top"]) * H)
    if fade is not None:
        y = np.linspace(0.0, 1.0, H)[:, None, None]
        f0, f1 = fade
        layer *= np.clip((f1 - y) / (f1 - f0), 0.0, 1.0) ** 1.5
    return add_glow(layer, 0.6)


# ── attractor ────────────────────────────────────────────────────────────────

def attractor_layers(H, W, box, supersample, split=None, exposure=3.0):
    theta = np.fromfile(TRAJECTORY, dtype=np.float64)
    lags = [int(round(l / RECORD_DT)) for l in LAGS]
    counts, nearness = accumulate(theta, lags, rotation(AZ, EL), W * supersample, H * supersample, box,
                                  substeps=4, split=split)
    return emissive(counts, nearness, PALETTE, gamma=0.9, exposure=exposure, bloom=0.55, supersample=supersample)


# ── layouts ──────────────────────────────────────────────────────────────────

# Tier sizes are fractions of the (square) canvas height.
FAR = {"n": 28, "height": (0.38, 0.55), "ramp": "far", "brightness": 0.55, "blur": 5.0, "top": (0.0, 0.06)}
MID = {"n": 18, "height": (0.50, 0.70), "ramp": "mid", "brightness": 0.75, "blur": 2.0, "top": (-0.02, 0.05)}
NEAR = {"n": 8, "height": (0.65, 0.90), "ramp": "near", "brightness": 0.9, "blur": 0.0, "top": (-0.03, 0.03),
        "curated": True}


def layout_curtain(H, W, seedlings, rng, ss):
    roots = curtain(H, W, seedlings, rng, [FAR, MID, NEAR])
    (att,) = attractor_layers(H, W, (0.12, 0.12, 0.05, 0.09), ss)
    img = screen(background(H, W, PALETTE, centre_y=0.45), roots)
    # A dark halo behind the ribbon keeps it legible over the roots.
    halo = gaussian_filter(att.max(axis=2), 0.02 * H)
    img *= (1.0 - 0.55 * np.clip(halo / (halo.max() + 1e-9) * 2.5, 0.0, 1.0))[..., None]
    return screen(img, att)


def layout_hero(H, W, seedlings, rng, ss, hero=HERO, box=(0.22, 0.03, 0.04, 0.04), hero_x=0.22, hero_h=1.1,
                hero_top=0.02):
    by_id = {s["id"]: s for s in seedlings}
    faint = [dict(FAR, n=20, brightness=0.45), dict(MID, n=10, brightness=0.5, blur=3.5)]
    roots = curtain(H, W, seedlings, rng, faint)

    rgb, alpha = render_seedling(by_id[hero], hero_h * H, ROOT_RAMPS["hero"], 1.0, 0.0, False, gamma=0.7)
    hero_rgb = np.zeros((H, W, 3))
    hero_a = np.zeros((H, W))
    paste(hero_rgb, rgb, hero_x * W, hero_top * H)
    paste(hero_a, alpha, hero_x * W, hero_top * H)
    hero_rgb = add_glow(hero_rgb, 0.5)

    back, front = attractor_layers(H, W, box, ss, split=0.5)
    img = screen(background(H, W, PALETTE), roots)
    img = screen(img, back)
    img = img * (1.0 - 0.85 * hero_a[..., None]) + hero_rgb  # the root hides what is behind it
    # the front half of the ribbon passes over the root
    front_a = np.clip(gaussian_filter(front.max(axis=2), 1.5) * 3.0, 0.0, 1.0)
    img = img * (1.0 - 0.5 * front_a[..., None])
    return screen(img, front)


def layout_plate(H, W, seedlings, rng, ss, n=6, top=0.04, height=0.85):
    """A row of seedlings as on the agar plate, in front of the glowing ribbon."""
    tall = [s for s in seedlings if s["id"] in CURATED]
    picks = rng.choice(len(tall), n, replace=False)
    row_rgb, row_a = np.zeros((H, W, 3)), np.zeros((H, W))
    for j, k in enumerate(picks):
        rgb, alpha = render_seedling(tall[k], height * H * rng.uniform(0.85, 1.0), ROOT_RAMPS["hero"], 0.95, 0.0,
                                     rng.random() < 0.5, gamma=0.7)
        x = (j + 0.5 + rng.uniform(-0.12, 0.12)) / n * W
        y = (top + rng.uniform(-0.01, 0.01)) * H
        paste(row_rgb, rgb, x, y)
        paste(row_a, alpha, x, y)
    row_rgb = add_glow(row_rgb, 0.6)
    roots = curtain(H, W, seedlings, rng, [dict(FAR, n=22, brightness=0.4, top=(0.0, 0.05))])

    (att,) = attractor_layers(H, W, (0.12, 0.12, 0.10, 0.04), ss, exposure=2.2)
    img = screen(screen(background(H, W, PALETTE), roots), 0.8 * att)
    img = img * (1.0 - 0.9 * row_a[..., None]) + row_rgb
    return img


def layout_descent(H, W, seedlings, rng, ss):
    tiers = [dict(FAR, n=32, height=(0.30, 0.42), top=(0.0, 0.04)),
             dict(MID, n=19, height=(0.36, 0.48), top=(-0.01, 0.03)),
             dict(NEAR, n=9, height=(0.42, 0.55), top=(-0.02, 0.02))]
    roots = curtain(H, W, seedlings, rng, tiers, fade=(0.20, 0.50))
    (att,) = attractor_layers(H, W, (0.15, 0.15, 0.28, 0.02), ss)
    return screen(screen(background(H, W, PALETTE, centre_y=0.65), roots), att)


LAYOUTS = {
    "curtain": layout_curtain,
    "plate": layout_plate,
    "hero": layout_hero,
    "hero2": lambda H, W, seedlings, rng, ss: layout_hero(H, W, seedlings, rng, ss, hero=HERO_ALT, hero_x=0.24,
                                                          hero_h=1.0),
    "descent": layout_descent,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=list(LAYOUTS))
    ap.add_argument("--size-in", type=float, default=8.0, help="side of the square cover")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    seedlings = load_seedlings()
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.layouts:
        rng = np.random.default_rng(args.seed)
        w_in = h_in = args.size_in
        W = H = int(round(args.size_in * args.dpi))
        img = LAYOUTS[name](H, W, seedlings, rng, args.supersample)
        base = os.path.join(
            args.out_dir,
            f"cover_{name}_tauk_{TAUK:g}_az_{AZ:g}_el_{EL:g}_{PALETTE}_seed_{args.seed}"
            f"_{w_in:g}x{h_in:g}in_{args.dpi}dpi",
        )
        save_image(np.clip(img, 0.0, 1.0), base, w_in, h_in, args.dpi)
        if args.dpi < 150:
            print(f"Saved: {base}.png")


if __name__ == "__main__":
    main()
