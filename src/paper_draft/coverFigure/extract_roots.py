"""Cut individual Arabidopsis seedlings out of the plate photographs.

Each segmented image in data/segmented and data/col_segmented/edited is split
into connected components; every component tall enough to be a whole seedling
is cropped from the raw photograph together with its mask.  The crops keep the
real photographic texture (root hairs, translucency), and the mask lets the
cover renderer drop the agar, the plate edge and the handwriting.

Output: outputs/root_crops.npz with, per seedling i,
    gray_i   uint8 crop of the raw photograph (grayscale)
    mask_i   uint8 crop of the segmentation mask (0/255)
and the source image of each crop in `source`.
"""

import glob
import os

import numpy as np
from PIL import Image
from skimage import measure

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "data"))
OUT_PATH = os.path.join(SCRIPT_DIR, "outputs", "root_crops.npz")

MIN_HEIGHT_FRAC = 0.30  # of the plate height: whole seedlings only
MAX_WIDTH_FRAC = 0.6    # of the component height: drops merged clumps
PAD = 6


def raw_for(segmented_path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(segmented_path))[0]
    for d in ("raw", "col_raw"):
        hits = sorted(glob.glob(os.path.join(DATA_DIR, d, stem + ".*")))
        if hits:
            return hits[0]
    return None


def main() -> None:
    seg_files = sorted(glob.glob(os.path.join(DATA_DIR, "segmented", "*.png")))
    seg_files += sorted(glob.glob(os.path.join(DATA_DIR, "col_segmented", "edited", "*.png")))

    out, sources = {}, []
    for seg_path in seg_files:
        raw_path = raw_for(seg_path)
        if raw_path is None:
            continue
        rgba = np.array(Image.open(seg_path))
        mask = rgba[..., 3] > 0 if rgba.shape[2] == 4 else rgba[..., :3].max(-1) > 0
        gray = np.array(Image.open(raw_path).convert("L"))
        if gray.shape != mask.shape:
            continue

        h_img = mask.shape[0]
        labeled = measure.label(mask, connectivity=2)
        for region in measure.regionprops(labeled):
            r0, c0, r1, c1 = region.bbox
            if r1 - r0 < MIN_HEIGHT_FRAC * h_img or (c1 - c0) > MAX_WIDTH_FRAC * (r1 - r0):
                continue
            r0, c0 = max(r0 - PAD, 0), max(c0 - PAD, 0)
            r1, c1 = min(r1 + PAD, h_img), min(c1 + PAD, mask.shape[1])
            i = len(sources)
            out[f"gray_{i}"] = gray[r0:r1, c0:c1]
            out[f"mask_{i}"] = (labeled[r0:r1, c0:c1] == region.label).astype(np.uint8) * 255
            sources.append(os.path.basename(seg_path))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, source=np.array(sources), **out)
    print(f"{len(sources)} seedlings from {len(set(sources))} plates -> {OUT_PATH}")


if __name__ == "__main__":
    main()
