"""Flyer surface reconstruction plots (heatmap / 3-D) at a given point in time."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..analysis.surface import compute_relative_surface

_LABEL_FONTSIZE = 14
_TICK_FONTSIZE = 12
_TITLE_FONTSIZE = 14
_SUPTITLE_FONTSIZE = 16
_CBAR_LABEL = "Relative displacement (µm)"


def _nearest_index(time_arr, t):
    idx = np.searchsorted(time_arr, t)
    if idx <= 0:
        return 0
    if idx >= len(time_arr):
        return len(time_arr) - 1
    before = idx - 1
    return idx if abs(time_arr[idx] - t) < abs(time_arr[before] - t) else before


def plot_flyer_surface_heatmap(data, positions_csv, t, reference_probe=None,
                                focus_scale=2.0, grid_size=300, ax=None):
    """2-D color-mesh reconstruction of the flyer surface at time ``t``.

    Interpolates each probe's displacement (relative to ``reference_probe``,
    or the mean if not given) onto a grid and draws it as a diverging
    heatmap, with the raw per-probe values overlaid as outlined points. See
    ``mpdv_toolbox.analysis.surface.compute_relative_surface`` for the
    parameters shared with the 3-D version.
    """
    x, y, z, Xi, Yi, Zi, t_actual = compute_relative_surface(
        data, positions_csv, t, reference_probe=reference_probe,
        focus_scale=focus_scale, grid_size=grid_size)

    clim = np.nanmax(np.abs(Zi))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    mesh = ax.pcolormesh(Xi, Yi, Zi, cmap="RdBu_r", vmin=-clim, vmax=clim, shading="auto")
    ax.scatter(x, y, c=z, cmap="RdBu_r", vmin=-clim, vmax=clim,
               edgecolors="k", linewidths=0.8, s=80, zorder=5)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(_CBAR_LABEL, fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    ax.set_xlabel("x position (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("y position (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_title(f"Flyer Surface at t = {t_actual * 1e9:.1f} ns", fontsize=_TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE)

    fig.tight_layout()
    return fig, ax


def plot_flyer_surface_3d(data, positions_csv, t, reference_probe=None,
                           focus_scale=2.0, grid_size=300, elev=20, azim=45, ax=None):
    """3-D surface reconstruction of the flyer at time ``t``.

    Interpolates each probe's displacement (relative to ``reference_probe``,
    or the mean if not given) onto a grid and draws it as a 3-D surface, with
    the raw per-probe values overlaid as outlined points. See
    ``mpdv_toolbox.analysis.surface.compute_relative_surface`` for the
    parameters shared with the heatmap version.
    """
    x, y, z, Xi, Yi, Zi, t_actual = compute_relative_surface(
        data, positions_csv, t, reference_probe=reference_probe,
        focus_scale=focus_scale, grid_size=grid_size)

    clim = np.nanmax(np.abs(Zi))

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    surf = ax.plot_surface(Xi, Yi, Zi, cmap="RdBu_r", vmin=-clim, vmax=clim,
                            alpha=0.85, edgecolor="none")
    ax.scatter(x, y, z, c=z, cmap="RdBu_r", vmin=-clim, vmax=clim,
               edgecolors="k", linewidths=0.5, s=60, zorder=5)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5)
    cbar.set_label(_CBAR_LABEL, fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("y (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_zlabel("Δz (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_title(f"Flyer Surface at t = {t_actual * 1e9:.1f} ns", fontsize=_TITLE_FONTSIZE)

    fig.tight_layout()
    return fig, ax


def plot_flyer_surface_evolution(data, positions_csv, timestep, reference_probe=None,
                                  focus_scale=2.0, grid_size=60, figsize=(13, 6)):
    """Publication figure showing flyer surface curvature evolving over flight.

    Two side-by-side panels -- a tilt-contour map and a 3-D wireframe stack --
    each overlay one snapshot per time step (spaced by ``timestep``) in a
    single axes, coloured by time, so the evolving curvature reads directly
    off one plot rather than needing a separate figure per moment (mirrors
    ``mpdv_toolbox.plotting.radial.plot_radial_displacement``'s ``timestep``
    spacing and single-axes overlay). Pick ``timestep`` so only a handful of
    snapshots are drawn (~5-10 is a good target) -- too fine a spacing makes
    either panel busy.

    data : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.process.process_pdv``.
    positions_csv : str
        Path to the probe-position CSV.
    timestep : float
        Spacing (seconds, same units as ``data["time"]``) between snapshots,
        spanning the full range of ``data["time"]``.
    reference_probe : int, optional
        Probe number to zero displacement against. Defaults to the mean.
    grid_size : int, optional
        Interpolation grid resolution per axis (default 60) -- coarser than
        the single-snapshot plots since many layers are drawn at once.

    Returns
    -------
    fig, (ax_contour, ax_3d)
    """
    time_arr = data["time"].values
    t_min, t_max = time_arr.min(), time_arr.max()
    requested = np.arange(t_min, t_max + timestep / 2, timestep)
    indices = sorted({_nearest_index(time_arr, t) for t in requested})

    surfaces = []
    for i in indices:
        try:
            _, _, _, Xi, Yi, Zi, t_actual = compute_relative_surface(
                data, positions_csv, time_arr[i], reference_probe=reference_probe,
                focus_scale=focus_scale, grid_size=grid_size)
        except ValueError as e:
            print(f"Warning: skipping t = {time_arr[i] * 1e9:.1f} ns ({e})")
            continue
        surfaces.append((t_actual, Xi, Yi, Zi))

    if not surfaces:
        raise ValueError("No time steps had enough valid, non-collinear probe data "
                          "to interpolate a surface.")

    clim = np.nanmax([np.nanmax(np.abs(Zi)) for *_, Zi in surfaces])
    times_ns = [t * 1e9 for t, *_ in surfaces]

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(times_ns[0], times_ns[-1])
    levels = np.linspace(-clim, clim, 7)[1:-1]  # drop the extremes for a cleaner map

    fig = plt.figure(figsize=figsize)
    ax_c = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    # Very coarse mesh on purpose: the underlying shape is only constrained by a
    # handful of probes (smooth/low-order), so a dense wireframe would just add
    # visual noise once several time steps are stacked in the same axes.
    stride = max(grid_size // 4, 1)
    for t_actual, Xi, Yi, Zi in surfaces:
        color = cmap(norm(t_actual * 1e9))
        ax_c.contour(Xi, Yi, Zi, levels=levels, colors=[color], linewidths=1.1, alpha=0.85)
        ax_3d.plot_wireframe(Xi, Yi, Zi, color=color, linewidth=0.9, alpha=0.7,
                              rstride=stride, cstride=stride)

    ax_c.set_xlabel("x position (µm)", fontsize=_LABEL_FONTSIZE)
    ax_c.set_ylabel("y position (µm)", fontsize=_LABEL_FONTSIZE)
    ax_c.set_title("Tilt Contours", fontsize=_TITLE_FONTSIZE)
    ax_c.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE, direction="in", length=6)
    ax_c.set_aspect("equal")
    for spine in ax_c.spines.values():
        spine.set_linewidth(1.2)

    ax_3d.set_xlabel("x (µm)", fontsize=_LABEL_FONTSIZE)
    ax_3d.set_ylabel("y (µm)", fontsize=_LABEL_FONTSIZE)
    ax_3d.set_zlabel("Δz (µm)", fontsize=_LABEL_FONTSIZE)
    ax_3d.set_zlim(-clim, clim)
    ax_3d.set_title("Surface Curvature", fontsize=_TITLE_FONTSIZE)
    ax_3d.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_c, ax_3d], shrink=0.75, pad=0.03)
    cbar.set_label("Time (ns)", fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    fig.suptitle(f"Flyer Surface Evolution  (t = {times_ns[0]:.0f}-{times_ns[-1]:.0f} ns, "
                 f"Δt = {timestep * 1e9:.1f} ns)", fontsize=_SUPTITLE_FONTSIZE)

    return fig, (ax_c, ax_3d)
