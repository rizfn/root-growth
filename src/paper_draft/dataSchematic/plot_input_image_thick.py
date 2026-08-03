from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from imageio.v2 import imread
from skimage import measure
from skimage.segmentation import expand_labels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wavelength_pipeline import ROOT_PALETTE_HEX, build_colored_root_rgba, save_svg  # noqa: E402


def plot_input_image_thick(image_path: Path, output_dir: Path, thickness_px: float) -> None:
    """Plot the wavelength-pipeline input image with dilated (thicker) roots."""
    rgba = imread(image_path)
    mask = rgba[:, :, 3] > 0 if rgba.shape[2] == 4 else np.any(rgba[:, :, :3] > 0, axis=2)

    labeled = measure.label(mask, connectivity=2)
    labeled = expand_labels(labeled, distance=thickness_px)

    props = sorted(measure.regionprops(labeled), key=lambda r: r.centroid[1])
    palette_rgb = [tuple(mcolors.to_rgb(c)) for c in ROOT_PALETTE_HEX]
    root_colors = {
        int(region.label): palette_rgb[i % len(palette_rgb)]
        for i, region in enumerate(props)
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    fig.patch.set_alpha(0.0)
    ax.imshow(build_colored_root_rgba(labeled, root_colors))
    ax.set_axis_off()
    ax.set_facecolor("none")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    save_svg(fig, output_dir / f"01_input_image_thick_{thickness_px:g}px.svg")


def main() -> None:
    plot_input_image_thick(
        image_path=Path("data/col_segmented/edited/Col-0_20220630112924-0001_1.5AgarVertical.png"),
        output_dir=Path("src/paper_draft/dataSchematic/plots/wavelength_pipeline"),
        thickness_px=1.0,
    )


if __name__ == "__main__":
    main()
