from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from matplotlib import colors
import matplotlib as mpl
from skimage import measure, color


def read_image_grayscale(path):
    im = imageio.imread(path)
    # convert to grayscale if RGB
    if im.ndim == 3:
        im = color.rgb2gray(im)
        # rgb2gray returns float in [0,1]
        im = (im * 255).astype(np.uint8)
    elif im.dtype == np.float32 or im.dtype == np.float64:
        # floats may be in [0,1]
        if im.max() <= 1.0:
            im = (im * 255).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
    else:
        im = im.astype(np.uint8)
    return im


def segment_image(im, min_threshold=55, max_threshold=75, top_n=10, max_x_extent_ratio=0.75, min_yx_extent_ratio=0.3, min_x_extent=20, hole_fill_size=25):
    # mask within thresholds (handles grayscale)
    mask = (im >= min_threshold) & (im <= max_threshold)

    labeled_mask = measure.label(mask, connectivity=2)

    image_width = im.shape[1]
    labels_to_remove = set()
    for region in measure.regionprops(labeled_mask):
        minr, minc, maxr, maxc = region.bbox
        x_extent = maxc - minc
        y_extent = maxr - minr
        if x_extent <= min_x_extent or max_x_extent_ratio * image_width <= x_extent or y_extent <= min_yx_extent_ratio * x_extent:
        # if max_x_extent_ratio * image_width <= x_extent or y_extent <= min_yx_extent_ratio * x_extent:
            labels_to_remove.add(region.label)

    if labels_to_remove:
        for lbl in labels_to_remove:
            labeled_mask[labeled_mask == lbl] = 0

    regions = measure.regionprops(labeled_mask)
    regions = sorted(regions, key=lambda r: r.area, reverse=True)
    top_labels = [r.label for r in regions[:top_n]]

    relabeled = np.zeros_like(labeled_mask, dtype=np.int32)
    for i, lbl in enumerate(top_labels, start=1):
        relabeled[labeled_mask == lbl] = i

    return relabeled


def segment_image_slow(im, min_threshold=55, max_threshold=75, top_n=10,
                       max_x_extent_ratio=0.75, min_yx_extent_ratio=0.3,
                       min_x_extent=20, frame_frac=0.1, frame_thresh=0.75, hole_fill_size=25):
    """Slower, more conservative segmentation.

    This version applies the same intensity and extent filters as
    `segment_image` but also removes any connected component for
    which more than `frame_thresh` fraction of its pixels lie inside
    the outer `frame_frac` fraction of the image (top/bottom/left/right).

    Use this when you want to exclude clusters that live largely in the
    image border/frame. Returns a relabeled image (0..k).
    """
    # initial threshold mask + labeling
    mask = (im >= min_threshold) & (im <= max_threshold)
    labeled = measure.label(mask, connectivity=2)

    # frame margins
    h, w = im.shape[:2]
    top_border = int(frame_frac * h)
    bottom_border = h - top_border
    left_border = int(frame_frac * w)
    right_border = w - left_border

    # remove regions mostly inside the border frame
    labels_to_remove = set()
    for region in measure.regionprops(labeled):
        coords = region.coords  # (N, 2) rows,cols
        if coords.size == 0:
            continue
        in_frame = ((coords[:, 0] < top_border) | (coords[:, 0] >= bottom_border) |
                    (coords[:, 1] < left_border) | (coords[:, 1] >= right_border))
        frac_in_frame = np.count_nonzero(in_frame) / coords.shape[0]
        if frac_in_frame > frame_thresh:
            labels_to_remove.add(region.label)

    if labels_to_remove:
        for lbl in labels_to_remove:
            labeled[labeled == lbl] = 0

    # apply the same extent-based filtering as the fast function
    image_width = w
    labels_to_remove2 = set()
    for region in measure.regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        x_extent = maxc - minc
        y_extent = maxr - minr
        if x_extent <= min_x_extent or max_x_extent_ratio * image_width <= x_extent or y_extent <= min_yx_extent_ratio * x_extent:
            labels_to_remove2.add(region.label)

    if labels_to_remove2:
        for lbl in labels_to_remove2:
            labeled[labeled == lbl] = 0

    # Fill small background holes completely enclosed by a single segment
    if hole_fill_size and hole_fill_size > 0:
        h, w = im.shape[:2]
        background_mask = (labeled == 0)
        bg_labeled = measure.label(background_mask, connectivity=1) # 4-connectivity for background
        for bg_region in measure.regionprops(bg_labeled):
            if bg_region.area > hole_fill_size:
                continue
            coords = bg_region.coords
            # skip if touching image border (not enclosed)
            if np.any(coords[:,0] == 0) or np.any(coords[:,0] == h-1) or np.any(coords[:,1] == 0) or np.any(coords[:,1] == w-1):
                continue
            neighbor_labels = []
            for (r, c) in coords:
                r0 = max(0, r-1); r1 = min(h-1, r+1)
                c0 = max(0, c-1); c1 = min(w-1, c+1)
                neigh = labeled[r0:r1+1, c0:c1+1].ravel()
                nonzero = neigh[neigh > 0]
                if nonzero.size > 0:
                    neighbor_labels.append(nonzero)
            if not neighbor_labels:
                continue
            neighbor_labels = np.concatenate(neighbor_labels)
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            chosen = labels[np.argmax(counts)]
            labeled[coords[:,0], coords[:,1]] = chosen

    # keep only the top_n largest remaining components
    regions = measure.regionprops(labeled)
    regions = sorted(regions, key=lambda r: r.area, reverse=True)
    top_labels = [r.label for r in regions[:top_n]]

    relabeled = np.zeros_like(labeled, dtype=np.int32)
    for i, lbl in enumerate(top_labels, start=1):
        relabeled[labeled == lbl] = i

    return relabeled


def save_relabeled_png(relabeled, out_path, top_n):
    # create a discrete rainbow-like colormap and map each integer label to a color
    # background (label 0) will be black
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # determine actual labels present (0..n_labels)
    n_labels = int(relabeled.max())
    if n_labels <= 0:
        # nothing found; save a black image
        rgba_uint8 = np.zeros(relabeled.shape + (4,), dtype=np.uint8)
        imageio.imwrite(str(out_path), rgba_uint8)
        return

    cmap_name = 'rainbow'
    cmap = mpl.colormaps.get(cmap_name)
    # create n_labels+1 colors (0..n_labels). index 0 reserved for background
    steps = np.linspace(0.0, 1.0, n_labels + 1)
    colors_rgba = cmap(steps)
    # force background to transparant
    colors_rgba[0] = np.array([0.0, 0.0, 0.0, 0.0])

    # map relabeled integers to RGBA pixels via lookup
    rgba_img = colors_rgba[relabeled]
    rgba_uint8 = (rgba_img * 255).astype(np.uint8)

    imageio.imwrite(str(out_path), rgba_uint8)


def run_batch(input_dir, pattern, mapping, out_dir, min_threshold, max_threshold, default_top_n):
    p = Path(input_dir)
    files = sorted(p.rglob(pattern))
    if not files:
        print(f"No files found for pattern {pattern} under {input_dir}")
        return

    out_dir = Path(out_dir)
    for fp in files:
        fname = fp.name
        top_n = default_top_n
        if mapping and fname in mapping:
            try:
                top_n = int(mapping[fname])
            except Exception:
                print(f"Warning: invalid mapping value for {fname}, using default {default_top_n}")

        print(f"Processing {fname} with top_n={top_n}")
        im = read_image_grayscale(fp)
        # relabeled = segment_image(im, min_threshold=min_threshold, max_threshold=max_threshold, top_n=top_n, max_x_extent_ratio=0.75, min_yx_extent_ratio=0.3, min_x_extent=20)
        relabeled = segment_image_slow(im, min_threshold=min_threshold, max_threshold=max_threshold, top_n=top_n, max_x_extent_ratio=0.75, min_yx_extent_ratio=0.3, min_x_extent=20)

        out_path = out_dir / (fp.stem + '.png')
        save_relabeled_png(relabeled, out_path, top_n)


def main():
    # defaults that used to live in `run_batch_simple`
    input_dir = Path('data/raw')
    pattern = '*.bmp'
    out_dir = Path('data/segmented')
    min_threshold = 55
    max_threshold = 75
    default_top_n = 10

    root_number_data = {
        '003-Ag-0-s-250612-002.bmp': 3,
        '003-Ag-0-v-250612-001.bmp': 5,
        '007-Amel-1-s-250612-005.bmp': 10,
        '007-Amel-1-v-250612-006.bmp': 10,
        '014-Baa-1-s-250612-018.bmp': 10,
        '014-Baa-1-v-250612-017.bmp': 10,
        '020-Ber-0-s-250612-003.bmp': 10,
        '020-Ber-0-v-250612-004.bmp': 9,
        '031-Cevr-1-s-250612-021.bmp': 9,
        '031-Cevr-1-v-250612-022.bmp': 11,
        '045-El-0-s-250612-011.bmp': 10,
        '045-El-0-v-250612-012.bmp': 11,
        '057-Gel-s-250612-019.bmp': 10,
        '057-Gel-v-250612-020.bmp': 10,
        '059-Gifu-2-s-250612-016.bmp': 10,
        '059-Gifu-2-v-250612-015.bmp': 10,
        '065-Hh-0-s-250612-014.bmp': 10,
        '065-Hh-0-v-250612-013.bmp': 10,
        '079-Ko-2-s-250612-009.bmp': 10,
        '079-Ko-2-v-250612-010.bmp': 10,
        '101-Neo-6-s-241217-187.bmp': 10,
        '101-Neo-6-v-241217-186.bmp': 8,
        '102-Nok-3-s-241217-191.bmp': 1,
        '102-Nok-3-v-241217-190.bmp': 1,
        '103-Np-7-s-241217-193.bmp': 2,
        '103-Np-7-v-241217-192.bmp': 0,
        '104-Nw-0-s-241217-195.bmp': 9,
        '104-Nw-0-v-241217-194.bmp': 9,
        '105-Ob-0-s-241217-197.bmp': 8,
        '105-Ob-0-v-241217-196.bmp': 10,
        '106-Old-1-s-241217-199.bmp': 7,
        '106-Old-1-v-241217-198.bmp': 7,
        '107-Or-0-s-241223-200.bmp': 11,
        '107-Or-0-v-241223-201.bmp': 8,
        '108-Ove-0-s-241223-203.bmp': 6,
        '108-Ove-0-v-241223-202.bmp': 8,
        '109-Per-1-s-241223-204.bmp': 6,
        '109-Per-1-v-241223-205.bmp': 8,
        '110-P1-0-s-241223-207.bmp': 7,
        '110-P1-0-v-241223-206.bmp': 7,
        '111-Pla-0-s-241223-209.bmp': 10,
        '111-Pla-0-v-241223-208.bmp': 9,
        '112-PnA-17-s-241223-210.bmp': 7,
        '112-PnA-17-v-241223-211.bmp': 10,
        '113-Pog-0-s-241223-212.bmp': 3,
        '113-Pog-0-v-241223-213.bmp': 4,
        '114-Pu2-23-s-241223-215.bmp': 3,
        '114-Pu2-23-v-241223-214.bmp': 3,
        '115-Pu2-7-s-241223-217.bmp': 4,
        '115-Pu2-7-v-241223-216.bmp': 5,
        '116-Qar-8a-s-241223-218.bmp': 8,
        '116-Qar-8a-v-241223-219.bmp': 9,
        '117-Ra-0-s-241223-221.bmp': 7,
        '117-Ra-0-v-241223-220.bmp': 8,
        '119-Rd-0-s-241223-222.bmp': 8,
        '119-Rd-0-v-241223-223.bmp': 10,
        '120-Rld-1-s-250130-225.bmp': 9,
        '120-Rld-1-v-250130-224.bmp': 6,
        '121-Rome-1-s-250130-226.bmp': 7,
        '121-Rome-1-v-250130-227.bmp': 6,
        '123-RRs-10-s-250130-228.bmp': 9,
        '123-RRs-10-v-250130-229.bmp': 9,
        '124-RRs-7-s-250130-231.bmp': 5,
        '124-RRs-7-v-250130-230.bmp': 7,
        }

    run_batch(input_dir, pattern, root_number_data, out_dir, min_threshold, max_threshold, default_top_n)


if __name__ == '__main__':
    main()
