"""Arc length at which each successive oscillation happens, for every root.

The wavelength analysis in ../wavelengths takes the orientation series theta(s)
of a root and asks, through the autocorrelation, for one number: the arc length
Omega over which theta repeats itself.  That throws away everything about *where*
along the root each wave sits, and so cannot see a wavelength that changes as the
root grows.

Here the autocorrelation is dropped and the waves are counted directly.  Every
time the detrended theta(s) crosses its baseline the root has completed half an
oscillation, so the crossings, in order, are

    s_0 < s_1 < s_2 < ...   at oscillation index   n = 0, 1/2, 1, 3/2, ...

and s(n) is the arc length "clock reading" of the root against its own internal
oscillator.

The assumptions that turns that into a statement about growth are:

  * the delay tau, and so the period T of the oscillation, is a property of the
    plant -- the same for every root and constant in time.  n = t/T is then a
    clock, and ds/dn = v*T is the elongation rate v in mm per oscillation;
  * v is common to the roots of a plate but *not* constant in time: a root grows
    slowly for the first days and faster later.  Anything that changes along one
    root is therefore attributed to v, not to T;
  * neither holds exactly, because roots vary.  That variability is the spread
    between roots in every figure here, and it is the reason the result is quoted
    as a comparison within one root -- each root its own control -- rather than
    between them.

A straight s(n) means the root grew at a constant rate; curvature in s(n) means
the rate changed while the root was being recorded, and the local slope
s(n+1) - s(n) is exactly the local wavelength Omega(n).  This is the same Omega as
the autocorrelation gives, but resolved along the root instead of averaged over
it -- and since it is not constant, the single number the autocorrelation returns
is an average over whatever stretch of root happened to be in the picture.

One warning about reading the figures.  s(n) is a running sum of the crossing
intervals: it is monotone by construction, and each of its points carries the
noise of all the earlier ones, averaged.  It therefore always looks clean, and
none of that cleanliness is evidence for anything.  The same information
differentiated is the much noisier local Omega, and that -- with the within-root
early/late comparison -- is where the result should be judged.

Crossings are found with a hysteresis rule: the signal must swing past +h before
a downward crossing counts and past -h before an upward one does, with h a fixed
fraction of the robust amplitude of that root's own theta.  A single noisy pixel
therefore cannot manufacture an oscillation, while a genuine wave -- which always
overshoots by roughly its amplitude -- is never missed.  The baseline subtracted
before the crossings are taken is a moving average over a window several
wavelengths wide, which removes the slow drift of the growth direction (nonzero
and wandering for the tilted plates) without touching the waves themselves.

Root extraction -- skeletonise, take the main trunk, sample it every dx_pixels of
arc length, differentiate to get theta -- is imported unchanged from
../wavelengths/errorbars_filter.py, so both analyses see exactly the same roots.

Counting is checked against the autocorrelation it replaces: on the same roots
the two agree per-root with r ~ 0.73, and the first autocorrelation peak sits at
a median 0.93 of the counted mm per oscillation, i.e. slightly short, as a first
peak picked off a discrete lag axis should be.  The defaults below (smooth 1 mm,
baseline 15 mm, threshold 0.3) are where that agreement is best.

Because a threshold that quietly dropped the odd early wave would manufacture
exactly the trend being claimed, the trend is also measured by an estimator with
no crossings and no threshold in it at all: windowed_wavelength() below, the peak
of the Fourier spectrum of theta in a sliding 20 mm window.  It gives the same
late/early ratio, 1.37 tilted and 1.35 vertical, against 1.51 and 1.41 from the
counting.  The counted ratio is also unchanged by dropping the first two
crossings of every root, so it is not the seedling transient at the start of the
trace either.
"""

import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage import measure

_WAVELENGTHS_DIR = Path(__file__).resolve().parent.parent / "wavelengths"
if str(_WAVELENGTHS_DIR) not in sys.path:
    sys.path.insert(0, str(_WAVELENGTHS_DIR))

from errorbars_filter import (  # noqa: E402
    build_skeleton_graph,
    compute_theta_timeseries,
    find_main_trunk,
    set_plot_style,
    style_axes,
)
from mode_errorbars_and_kde import compute_half_sample_mode  # noqa: E402


COLOR_TILTED = "#901A1E"
COLOR_NON_TILTED = "#547AA5"
COLOR_REFERENCE = "#666666"
COLOR_YELLOW = "#CBA810"

# Plate height in the images is a fixed 100 mm, which sets the scale of each image.
PLATE_HEIGHT_MM = 100.0


def moving_average(y, window_samples):
    """Boxcar average of y over an odd window, reflected at both ends."""
    y = np.asarray(y, dtype=float)
    win = int(max(1, round(window_samples)))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return y.copy()
    if win >= len(y):
        return np.full_like(y, float(np.mean(y)))

    pad = win // 2
    padded = np.pad(y, pad, mode="reflect")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(padded, kernel, mode="valid")


def central(values, kind="median"):
    """One central value for a set of roots: their median, or their mode.

    Wavelengths across roots are not symmetric -- a handful of roots sit far
    above the bulk and none can sit below zero -- so a median, let alone a mean,
    is pulled up by the tail and reports a wavelength that no root actually has.
    The mode is the half-sample mode of ../wavelengths, the same estimator the
    autocorrelation analysis settled on, so the two are directly comparable.

    Within a single root the choice is not open: half a root gives five or ten
    intervals, far too few for a mode, and those are summarised by a median
    whichever variant is being drawn.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if kind == "mode":
        return float(compute_half_sample_mode(v))
    return float(np.median(v))


def central_by_column(values, kind="median"):
    """central() applied down each column of a NaN-padded array of curves."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([])
    return np.array([central(values[:, j], kind) for j in range(values.shape[1])])


def robust_amplitude(y):
    """MAD-based standard deviation, insensitive to a few wild samples."""
    y = np.asarray(y, dtype=float)
    mad = float(np.median(np.abs(y - np.median(y))))
    return 1.4826 * mad


def detrend_theta(s_mm, theta, smooth_mm=1.0, baseline_mm=15.0):
    """High-pass theta(s): light smoothing, then a wide moving-average baseline."""
    s_mm = np.asarray(s_mm, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if len(s_mm) < 2:
        return theta.copy()

    ds = float(np.median(np.diff(s_mm)))
    smoothed = moving_average(theta, smooth_mm / ds)
    baseline = moving_average(smoothed, baseline_mm / ds)
    return smoothed - baseline


def _interpolated_zero(s_mm, y, i0, i1):
    """Arc length of the first sign change of y between samples i0 and i1."""
    for j in range(i0, i1):
        y0, y1 = y[j], y[j + 1]
        if y0 == 0.0:
            return float(s_mm[j])
        if y0 * y1 < 0.0:
            t = -y0 / (y1 - y0)
            return float(s_mm[j] + t * (s_mm[j + 1] - s_mm[j]))
    return None


def detect_crossings(s_mm, y, rel_threshold=0.3):
    """Baseline crossings of y(s) with hysteresis.

    A crossing is only recorded once the signal has committed to the other side,
    i.e. has passed +/- rel_threshold * (robust amplitude of y).  Returns the arc
    lengths of the crossings and their directions (+1 upward, -1 downward), which
    strictly alternate.
    """
    s_mm = np.asarray(s_mm, dtype=float)
    y = np.asarray(y, dtype=float)

    amp = robust_amplitude(y)
    if not np.isfinite(amp) or amp <= 0:
        return np.array([]), np.array([], dtype=int)

    h = rel_threshold * amp
    state = 0
    last_idx = 0
    crossings, directions = [], []

    for i, value in enumerate(y):
        if value > h:
            side = 1
        elif value < -h:
            side = -1
        else:
            continue

        if state == 0:
            state, last_idx = side, i
            continue

        if side == state:
            last_idx = i
            continue

        s_cross = _interpolated_zero(s_mm, y, last_idx, i)
        if s_cross is not None:
            crossings.append(s_cross)
            directions.append(side)
        state, last_idx = side, i

    return np.asarray(crossings, dtype=float), np.asarray(directions, dtype=int)


def fit_growth(index, s_cross):
    """Straight-line fit of arc length against oscillation index.

    The slope is the root's mean mm per oscillation -- the same quantity the
    autocorrelation estimates, arrived at by counting.  It is only a summary: the
    whole point of the folder is that s(n) is not straight, so the number depends
    on how much of the root was in the image.
    """
    index = np.asarray(index, dtype=float)
    s_cross = np.asarray(s_cross, dtype=float)
    out = {"slope_mm_per_osc": np.nan, "slope_err_mm_per_osc": np.nan, "rms_residual_mm": np.nan}
    if len(index) < 3:
        return out

    coeffs, cov = np.polyfit(index, s_cross, 1, cov=True)
    out["slope_mm_per_osc"] = float(coeffs[0])
    out["slope_err_mm_per_osc"] = float(np.sqrt(cov[0, 0]))
    out["rms_residual_mm"] = float(np.sqrt(np.mean((s_cross - np.polyval(coeffs, index)) ** 2)))
    return out


def early_late_wavelengths(crossings_mm, min_crossings=7):
    """Median full oscillation over the first and the second half of one root.

    Aggregating across roots at fixed n is contaminated by which roots are still
    present at large n -- the short-wavelength ones, since a root of given length
    fits more of them in.  This comparison never leaves a single root, so nothing
    about the other roots can enter it.
    """
    s_cross = np.asarray(crossings_mm, dtype=float)
    if len(s_cross) < min_crossings:
        return np.nan, np.nan
    gaps = np.diff(s_cross)  # half oscillations
    half = len(gaps) // 2
    return 2.0 * float(np.median(gaps[:half])), 2.0 * float(np.median(gaps[-half:]))


def windowed_wavelength(s_mm, theta, window_mm=20.0, step_mm=2.5, lam_bounds=(2.0, 20.0)):
    """Local wavelength from the Fourier peak of theta in a sliding window.

    Deliberately shares nothing with the crossing counter -- no baseline, no
    threshold, no smoothing -- so that it can be used to check that the drift in
    Omega along a root is in the data and not in the detection rule.
    """
    s_mm = np.asarray(s_mm, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if len(theta) < 8:
        return np.array([]), np.array([])

    ds = float(np.median(np.diff(s_mm)))
    win = int(round(window_mm / ds))
    if win < 8 or win > len(theta):
        return np.array([]), np.array([])
    step = max(1, int(round(step_mm / ds)))

    y = theta - np.mean(theta)
    taper = np.hanning(win)
    n_fft = 8 * win
    freq = np.fft.rfftfreq(n_fft, d=ds)
    lam = np.divide(1.0, freq, out=np.full_like(freq, np.inf), where=freq > 0)
    in_band = (lam >= lam_bounds[0]) & (lam <= lam_bounds[1])

    centres, lambdas = [], []
    for i0 in range(0, len(y) - win + 1, step):
        seg = y[i0 : i0 + win]
        spec = np.abs(np.fft.rfft((seg - seg.mean()) * taper, n=n_fft)) ** 2
        if not in_band.any():
            continue
        centres.append(s_mm[i0 + win // 2])
        lambdas.append(lam[in_band][np.argmax(spec[in_band])])

    return np.asarray(centres, dtype=float), np.asarray(lambdas, dtype=float)


def analyze_image(
    filepath,
    dx_pixels=8,
    min_root_length=200,
    smooth_mm=1.0,
    baseline_mm=15.0,
    rel_threshold=0.3,
):
    """Per-root crossing arc lengths and growth fits for one segmented image."""
    print(f"Processing {filepath.name}")
    rgba = imageio.imread(filepath)
    mm_per_pixel = PLATE_HEIGHT_MM / rgba.shape[0]

    if rgba.shape[2] == 4:
        mask = rgba[:, :, 3] > 0
    else:
        mask = np.any(rgba[:, :, :3] > 0, axis=2)

    labeled = measure.label(mask, connectivity=2)

    rows = []
    for region in measure.regionprops(labeled):
        comp_mask = labeled == region.label

        graph, _, coords = build_skeleton_graph(comp_mask)
        if graph is None or len(coords) < 2:
            continue

        main_path = find_main_trunk(graph, coords)
        if len(main_path) < 2:
            continue

        pts_main = np.array(main_path, dtype=float)
        seg_d_main = np.sqrt(np.sum(np.diff(pts_main, axis=0) ** 2, axis=1))
        main_length = seg_d_main.sum() if seg_d_main.size > 0 else 0.0
        if main_length < min_root_length:
            continue

        s_px, thetas = compute_theta_timeseries(main_path, dx_pixels, start_offset=min_root_length)
        if thetas is None or len(thetas) < 12:
            continue

        s_mm = np.asarray(s_px, dtype=float) * mm_per_pixel
        detrended = detrend_theta(s_mm, thetas, smooth_mm=smooth_mm, baseline_mm=baseline_mm)
        crossings, directions = detect_crossings(s_mm, detrended, rel_threshold=rel_threshold)
        if len(crossings) < 3:
            continue

        # Half an oscillation between consecutive crossings; n = 0 at the first one.
        index = 0.5 * np.arange(len(crossings), dtype=float)
        fit = fit_growth(index, crossings)
        omega_early, omega_late = early_late_wavelengths(crossings)

        rows.append(
            {
                "image": filepath.name,
                "root_label": int(region.label),
                "mm_per_pixel": float(mm_per_pixel),
                "dx_mm": float(dx_pixels * mm_per_pixel),
                "main_length_mm": float(main_length * mm_per_pixel),
                "s_mm": s_mm.tolist(),
                "theta": np.asarray(thetas, dtype=float).tolist(),
                "theta_detrended": np.asarray(detrended, dtype=float).tolist(),
                "crossings_mm": crossings.tolist(),
                "crossing_dirs": directions.tolist(),
                "oscillation_index": index.tolist(),
                "n_oscillations": float(index[-1]),
                "amplitude_rad": float(robust_amplitude(detrended)),
                "omega_early_mm": omega_early,
                "omega_late_mm": omega_late,
                **fit,
            }
        )

    print(f"  Counted oscillations for {len(rows)} roots")
    return rows


def group_of(image_name):
    """Same split as the wavelength analysis: tilted plates are named AgarTilted."""
    return "Tilted" if "AgarTilted" in str(image_name) else "Vertical"


def split_groups(rows):
    return {
        "Tilted": [r for r in rows if group_of(r["image"]) == "Tilted"],
        "Vertical": [r for r in rows if group_of(r["image"]) == "Vertical"],
    }


def aligned_curves(rows):
    """Arc length measured from each root's own first crossing, on the index grid.

    Roots are cut off at both ends at an arbitrary phase of the oscillation, so
    the absolute arc length of crossing number zero carries no information; only
    the increments do.  Shifting every root to s(0) = 0 makes the curves
    comparable and is what the aggregate plots average over.
    """
    curves = []
    for row in rows:
        index = np.asarray(row["oscillation_index"], dtype=float)
        s_cross = np.asarray(row["crossings_mm"], dtype=float)
        curves.append((index, s_cross - s_cross[0]))
    return curves


def stack_on_index_grid(curves):
    """The aligned curves as one NaN-padded array on the half-integer index grid."""
    if not curves:
        return np.array([]), np.zeros((0, 0))

    n_max = max(len(idx) for idx, _ in curves)
    grid = 0.5 * np.arange(n_max, dtype=float)
    values = np.full((len(curves), n_max), np.nan, dtype=float)
    for i, (_, s_rel) in enumerate(curves):
        values[i, : len(s_rel)] = s_rel
    return grid, values


def local_wavelengths(row):
    """Arc length of each full oscillation, and the index at its midpoint.

    Uses crossings two apart, so each value is a complete period rather than a
    half period; consecutive values overlap by half an oscillation.
    """
    s_cross = np.asarray(row["crossings_mm"], dtype=float)
    index = np.asarray(row["oscillation_index"], dtype=float)
    if len(s_cross) < 3:
        return np.array([]), np.array([])
    return index[:-2] + 0.5, s_cross[2:] - s_cross[:-2]
