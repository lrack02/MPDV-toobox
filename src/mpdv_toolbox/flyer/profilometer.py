"""Loader and colormap plotting for Zygo ASCII ``.xyz`` profilometer scans.

Zygo's ASCII "XYZ Data File" export writes one ``x y z`` row per pixel
(``x``, ``y`` are integer pixel indices, ``z`` is the calibrated surface
height) between a pair of lone ``#`` delimiter lines, preceded by a header
block that (among other things) carries the lateral pixel size in meters.
Pixels with no valid fringe data are written as ``x y No Data`` instead of a
numeric height.

See ``load_zygo_xyz`` for the header assumptions and ``plot_profilometer_scan``
for the colormap.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from tqdm import tqdm

_LABEL_FONTSIZE = 14
_TICK_FONTSIZE = 12
_TITLE_FONTSIZE = 14


def load_zygo_xyz(filepath, show_progress=True):
    """Parse a Zygo ASCII "XYZ Data File - Format 1" profilometer scan.

    The data block (between the two lone ``#`` lines) is read row by row
    rather than with ``np.genfromtxt`` because missing pixels are written
    as the literal tokens ``No Data`` in place of a numeric height, which
    would otherwise make the file ragged.

    Parameters
    ----------
    filepath : str
        Path to the ``.xyz`` file.
    show_progress : bool, optional
        Show a ``tqdm`` progress bar while parsing the (typically ~1e6 line)
        data block. Default True.

    Returns
    -------
    height : np.ndarray, shape (ny, nx)
        Height map, ``height[y, x]``, in the file's native height units
        (for Zygo ASCII exports this is calibrated height, conventionally
        microns -- the header does not name the unit explicitly, so treat
        it as an assumption, not a certainty). Missing ("No Data") pixels
        are NaN.
    pixel_size_m : float or None
        Lateral size of one pixel in meters, read from the header's 8-field
        system line (``... camera_res timestamp``). None if that line
        could not be found.
    meta : dict
        Any other header fields recovered: ``"timestamp"`` (Unix seconds,
        if found) and ``"header_lines"`` (raw header text, for inspection).
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines or "Zygo XYZ Data File" not in lines[0]:
        raise ValueError(f"{filepath} does not look like a Zygo XYZ Data File")

    hash_positions = [i for i, line in enumerate(lines) if line.strip() == "#"]
    if len(hash_positions) < 2:
        raise ValueError(
            f"Expected two '#' delimiter lines bracketing the data block in {filepath}, "
            f"found {len(hash_positions)}"
        )
    data_start, data_end = hash_positions[0] + 1, hash_positions[1]
    header_lines = lines[:hash_positions[0]]
    data_lines = lines[data_start:data_end]

    # Lateral pixel size: Zygo's 8-field system line is
    # "intensity_avg num_avgs wavelength_in ... camera_res timestamp" -- camera_res
    # (meters/pixel) is second-to-last, timestamp last. Identify it by shape (8
    # numeric tokens) plus a sanity range on camera_res, rather than a hardcoded
    # line index, in case unrelated fields shift between scans.
    pixel_size_m = None
    timestamp = None
    for line in header_lines:
        tokens = line.split()
        if len(tokens) != 8:
            continue
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            continue
        candidate = values[-2]
        if 1e-9 < candidate < 1e-3:
            pixel_size_m = candidate
            timestamp = values[-1]
            break

    n = len(data_lines)
    xs = np.empty(n, dtype=np.int64)
    ys = np.empty(n, dtype=np.int64)
    zs = np.empty(n, dtype=np.float64)

    iterator = tqdm(data_lines, desc="Parsing xyz data", unit="pts") if show_progress else data_lines
    for i, line in enumerate(iterator):
        parts = line.split()
        xs[i] = int(parts[0])
        ys[i] = int(parts[1])
        try:
            zs[i] = float(parts[2])
        except ValueError:
            zs[i] = np.nan  # "No Data" pixel

    nx, ny = xs.max() + 1, ys.max() + 1
    height = np.full((ny, nx), np.nan)
    height[ys, xs] = zs

    n_missing = np.isnan(height).sum()
    if n_missing:
        print(f"{filepath}: {n_missing}/{height.size} pixels ({100 * n_missing / height.size:.1f}%) "
              f"are 'No Data' and shown as gaps")

    meta = {"timestamp": timestamp, "header_lines": header_lines}
    return height, pixel_size_m, meta


def _nice_length(value):
    """Round ``value`` down to a "nice" 1/2/5 x 10^n number for a scale bar."""
    if value <= 0:
        return value
    exponent = np.floor(np.log10(value))
    fraction = value / 10 ** exponent
    if fraction >= 5:
        nice = 5
    elif fraction >= 2:
        nice = 2
    else:
        nice = 1
    return nice * 10 ** exponent


def _add_scale_bar(ax, bar_length, units, frac_x=0.95, frac_y=0.08, color="white"):
    """Draw a horizontal scale bar of physical length ``bar_length`` (data units)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_anchor = x0 + frac_x * (x1 - x0)
    y_anchor = y0 + frac_y * (y1 - y0)
    height_frac = 0.015 * abs(y1 - y0)

    ax.add_patch(Rectangle(
        (x_anchor - bar_length, y_anchor), bar_length, height_frac,
        facecolor=color, edgecolor="k", linewidth=0.8, zorder=5))
    ax.text(x_anchor - bar_length / 2, y_anchor + height_frac + 0.015 * abs(y1 - y0),
            f"{bar_length:g} {units}", color=color, ha="center", va="bottom",
            fontsize=_TICK_FONTSIZE, zorder=5,
            path_effects=[withStroke(linewidth=2, foreground="k")])


def remove_tilt(height, sigma_clip=3.0, n_iter=5):
    """Subtract a robustly-fit tilted plane from a height map.

    Fits ``z = a*x + b*y + c`` (pixel-index coordinates) by iteratively
    reweighted least squares: fit to all currently-kept valid pixels, then
    drop pixels whose residual exceeds ``sigma_clip`` standard deviations
    and refit, repeating for up to ``n_iter`` rounds (stopping early once
    the kept set stops changing). Raised or recessed features (e.g. the
    flyer's raised ring) are a minority of the scan area and end up
    clipped as outliers, so the plane converges to the flat background --
    no explicit geometric segmentation of "inside" vs. the feature is
    needed, and it degrades gracefully to an ordinary plane fit if the
    whole scan is actually flat.

    Parameters
    ----------
    height : np.ndarray, shape (ny, nx)
        Height map, NaN where data is missing.
    sigma_clip : float, optional
        Residual threshold (in standard deviations of the current fit) for
        excluding a pixel from the next iteration. Default 3.0.
    n_iter : int, optional
        Maximum number of fit/clip rounds. Default 5.

    Returns
    -------
    corrected : np.ndarray, shape (ny, nx)
        ``height`` minus the fitted plane. NaNs are preserved.
    plane : np.ndarray, shape (ny, nx)
        The fitted plane, evaluated over the full grid.
    inlier_mask : np.ndarray, shape (ny, nx), bool
        Which valid pixels were kept in the final fit (False for both
        NaN pixels and clipped outliers) -- e.g. the raised ring should
        come back False here.
    """
    ny, nx = height.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    valid = np.isfinite(height)
    xs, ys, zs = xx[valid].astype(float), yy[valid].astype(float), height[valid]

    keep = np.ones(zs.shape, dtype=bool)
    for _ in range(n_iter):
        A = np.column_stack([xs[keep], ys[keep], np.ones(keep.sum())])
        coeffs, *_ = np.linalg.lstsq(A, zs[keep], rcond=None)
        residuals = zs - (coeffs[0] * xs + coeffs[1] * ys + coeffs[2])
        # Median absolute deviation rather than std: a raised/recessed feature
        # covering a sizeable minority of the area (e.g. the ring) inflates a
        # plain std enough that it barely gets clipped; MAD stays robust as
        # long as outliers are under ~50% of the kept points.
        resid_kept = residuals[keep]
        mad = np.median(np.abs(resid_kept - np.median(resid_kept)))
        robust_sigma = 1.4826 * mad
        new_keep = np.abs(residuals) < sigma_clip * robust_sigma
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    plane = coeffs[0] * xx.astype(float) + coeffs[1] * yy.astype(float) + coeffs[2]
    corrected = height - plane

    inlier_mask = np.zeros_like(valid)
    inlier_mask[valid] = keep
    return corrected, plane, inlier_mask


def plot_profilometer_scan(height, pixel_size_m=None, zero_quantile=0.01,
                            lateral_units="µm", height_units="µm",
                            clip_percentiles=(0.5, 99.5), cmap="viridis",
                            ax=None, title=None):
    """Plot a profilometer height map as a colormap with a scale bar.

    Parameters
    ----------
    height : np.ndarray, shape (ny, nx)
        Height map (e.g. from ``load_zygo_xyz``), NaN where data is missing.
    pixel_size_m : float, optional
        Lateral size of one pixel, in meters (e.g. from ``load_zygo_xyz``).
        If given, axes are drawn in physical units (``lateral_units``) and a
        scale bar is added; if None, axes are left in pixel index units and
        no scale bar is drawn.
    zero_quantile : float, optional
        Quantile (0-1) of the valid height values used as the zero/baseline
        level, in place of the raw minimum -- robust to a handful of
        low-outlier pixels setting an unrepresentative zero. Default 0.01
        (1st percentile). Use 0.5 to zero to the median, etc.
    lateral_units : str, optional
        Unit label for the x/y axes and scale bar when ``pixel_size_m`` is
        given (``pixel_size_m`` is assumed to be in meters). Default "µm".
    height_units : str, optional
        Unit label for the colorbar. The height array's values are used
        as-is -- no unit conversion is applied. Default "µm".
    clip_percentiles : (float, float), optional
        Lower/upper percentile of the zeroed heights used to set the
        colormap's vmin/vmax, so a few extreme pixels don't wash out the
        contrast over the bulk of the surface. Default (0.5, 99.5). Pass
        ``(0, 100)`` to span the full range instead.
    cmap : str, optional
        Matplotlib colormap name. Default "viridis".
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure/axes is created if not given.
    title : str, optional
        Axes title.

    Returns
    -------
    fig, ax, zeroed : figure, axes, np.ndarray
        The figure, axes, and the baseline-zeroed height map that was
        plotted.
    """
    baseline = np.nanquantile(height, zero_quantile)
    zeroed = height - baseline

    vmin, vmax = np.nanpercentile(zeroed, clip_percentiles)

    ny, nx = height.shape
    if pixel_size_m is not None:
        scale = pixel_size_m * 1e6 if lateral_units in ("µm", "um") else pixel_size_m
        extent = (0, nx * scale, 0, ny * scale)
    else:
        extent = (0, nx, 0, ny)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(zeroed, cmap=cmap, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Height ({height_units})",
                    fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    if pixel_size_m is not None:
        field_width = nx * scale
        _add_scale_bar(ax, _nice_length(0.2 * field_width), lateral_units)
        ax.set_xlabel(f"x ({lateral_units})", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(f"y ({lateral_units})", fontsize=_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("x (pixel)", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel("y (pixel)", fontsize=_LABEL_FONTSIZE)

    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE)

    fig.tight_layout()
    return fig, ax, zeroed


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    xyz_path = repo_root / "F870.xyz"
    out_path = repo_root / "figures" / "F870_profilometer.png"

    height, pixel_size_m, meta = load_zygo_xyz(xyz_path)
    corrected, plane, inlier_mask = remove_tilt(height)
    print(f"Tilt fit used {inlier_mask.sum()}/{np.isfinite(height).sum()} pixels as background "
          f"after clipping outliers (e.g. the raised ring)")

    fig, ax, zeroed = plot_profilometer_scan(
        corrected, pixel_size_m=pixel_size_m, zero_quantile=0.01,
        title=xyz_path.stem)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")

