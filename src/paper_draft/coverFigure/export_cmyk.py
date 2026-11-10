"""Print-ready (four-colour, CMYK) version of a rendered cover.

The cover is rendered in sRGB (compose_cover.py).  Here it is separated into
CMYK through an ICC output profile, by default Coated FOGRA39 (ISO 12647-2
coated paper, the European offset standard), with perceptual intent and black
point compensation so that the saturated purples and magentas that lie outside
the CMYK gamut are compressed smoothly rather than clipped.

Outputs, next to the input:
    <name>_CMYK_<profile>.tif   Deflate-compressed (lossless) CMYK TIFF with the
                                profile embedded; about half the size of LZW
    <name>_CMYK_<profile>.pdf   the same separation as a PDF (JPEG, quality 95)
    <name>_CMYK_<profile>_softproof.png
                                the CMYK file converted back to sRGB: what the
                                printed cover will look like on screen

and a short report of how far the printed colours move from the screen
colours (CIE76 Delta E) and of the maximum total ink coverage.
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageCms

Image.MAX_IMAGE_PIXELS = None

ADOBE_PROFILES = "/mnt/c/Program Files (x86)/Common Files/Adobe/Color/Profiles/Recommended"
DEFAULT_PROFILE = os.path.join(ADOBE_PROFILES, "CoatedFOGRA39.icc")


def lab(img: Image.Image, src: ImageCms.ImageCmsProfile, mode: str) -> np.ndarray:
    t = ImageCms.buildTransform(src, ImageCms.createProfile("LAB"), mode, "LAB",
                                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    a = np.asarray(ImageCms.applyTransform(img, t))
    # PIL stores L scaled to 0..255, and a, b as signed bytes in an unsigned container
    ab = a[..., 1:].astype(np.int8).astype(float)
    return np.stack([a[..., 0] * 100.0 / 255.0, ab[..., 0], ab[..., 1]], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="sRGB PNG from compose_cover.py")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, help="CMYK output ICC profile")
    ap.add_argument("--dpi", type=int, required=True)
    args = ap.parse_args()

    rgb = Image.open(args.input).convert("RGB")
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    cmyk_profile = ImageCms.getOpenProfile(args.profile)
    to_cmyk = ImageCms.buildTransform(srgb, cmyk_profile, "RGB", "CMYK",
                                      renderingIntent=ImageCms.Intent.PERCEPTUAL,
                                      flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    cmyk = ImageCms.applyTransform(rgb, to_cmyk)

    tag = os.path.splitext(os.path.basename(args.profile))[0]
    base = os.path.splitext(args.input)[0] + f"_CMYK_{tag}"
    cmyk.save(base + ".tif", compression="tiff_adobe_deflate", dpi=(args.dpi, args.dpi),
              icc_profile=cmyk_profile.tobytes())
    cmyk.save(base + ".pdf", "PDF", resolution=args.dpi, quality=95)

    # Black point compensation in the proof as well: it shows the print as the
    # eye sees it once adapted to the paper, not the paper's black on a screen.
    to_rgb = ImageCms.buildTransform(cmyk_profile, srgb, "CMYK", "RGB",
                                     renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                     flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    proof = ImageCms.applyTransform(cmyk, to_rgb)
    proof.save(base + "_softproof.png", dpi=(args.dpi, args.dpi))

    lab_screen, lab_print = lab(rgb, srgb, "RGB"), lab(cmyk, cmyk_profile, "CMYK")
    delta_e = np.linalg.norm(lab_screen - lab_print, axis=-1)
    ink = np.asarray(cmyk, dtype=float).sum(axis=-1) / 255.0 * 100.0
    print(f"Saved: {base}.tif / .pdf / _softproof.png  ({cmyk.size[0]}x{cmyk.size[1]} px, {args.dpi} dpi)")
    # The near-black ground is darker than any ink on paper, so it is reported
    # on its own; the rest is the ribbon and the seedlings.
    ground = lab_screen[..., 0] < 12.0
    print(f"Ground (L* < 12, {100 * ground.mean():.0f}% of the page): L* {np.median(lab_screen[ground, 0]):.1f} "
          f"on screen -> {np.median(lab_print[ground, 0]):.1f} in print")
    feat = delta_e[~ground]
    print(f"Ribbon and seedlings, Delta E screen -> print: median {np.median(feat):.1f}, "
          f"95th pct {np.percentile(feat, 95):.1f}, max {feat.max():.1f}")
    print(f"Total ink coverage: max {ink.max():.0f}%, 99.9th pct {np.percentile(ink, 99.9):.0f}%")


if __name__ == "__main__":
    main()
