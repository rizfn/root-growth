from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io, measure


def load_grayscale_uint8(image_path: Path) -> np.ndarray:
    image = io.imread(str(image_path))
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        return (image * 255).astype(np.uint8)

    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def threshold_image(grayscale: np.ndarray, low: int, high: int) -> np.ndarray:
    return (grayscale >= low) & (grayscale <= high)


def relabel_top_components(mask: np.ndarray, top_n: int) -> np.ndarray:
    labeled = measure.label(mask, connectivity=2)
    image_width = mask.shape[1]

    labels_to_remove: set[int] = set()
    for region in measure.regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        x_extent = maxc - minc
        y_extent = maxr - minr
        if x_extent >= 0.75 * image_width or y_extent <= 0.3 * x_extent or x_extent <= 20:
            labels_to_remove.add(region.label)

    if labels_to_remove:
        removal_mask = np.isin(labeled, list(labels_to_remove))
        labeled[removal_mask] = 0

    regions = sorted(measure.regionprops(labeled), key=lambda r: r.area, reverse=True)
    top_labels = [region.label for region in regions[:top_n]]

    relabeled = np.zeros_like(labeled, dtype=np.int32)
    for i, label_id in enumerate(top_labels, start=1):
        relabeled[labeled == label_id] = i

    return relabeled


def save_stage_svg(
    image: np.ndarray,
    output_path: Path,
    cmap: str,
    norm: mcolors.Normalize | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    ax.imshow(image, cmap=cmap, norm=norm)
    ax.set_axis_off()
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def run_pipeline(image_path: Path, output_dir: Path, threshold_low: int, threshold_high: int, top_n: int) -> None:
    grayscale = load_grayscale_uint8(image_path)
    mask = threshold_image(grayscale, threshold_low, threshold_high)
    relabeled = relabel_top_components(mask, top_n=top_n)

    save_stage_svg(
        image=grayscale,
        output_path=output_dir / "01_grayscale.svg",
        cmap="gray",
    )

    save_stage_svg(
        image=mask.astype(np.uint8),
        output_path=output_dir / "02_threshold_mask.svg",
        cmap="gray",
        norm=mcolors.Normalize(vmin=0, vmax=1),
    )

    n_labels = max(int(relabeled.max()), top_n)
    cmap = plt.get_cmap("nipy_spectral", n_labels + 1)
    boundaries = np.arange(n_labels + 2) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    save_stage_svg(
        image=relabeled,
        output_path=output_dir / "03_segmented_components.svg",
        cmap=cmap,
        norm=norm,
    )


def main() -> None:
    run_pipeline(
        image_path=Path("data/col_raw/Col-0_20220630112924-0001_1.5AgarVertical.tif"),
        output_dir=Path("src/paper_draft/dataSchematic/plots/segmentation_pipeline"),
        threshold_low=85,
        threshold_high=150,
        top_n=8,
    )


if __name__ == "__main__":
    main()
