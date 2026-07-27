"""Flyer surface reconstruction plots (heatmap / 3-D) at a given point in time."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..analysis.surface import compute_relative_surface

_LABEL_FONTSIZE = 14
_TICK_FONTSIZE = 12
_TITLE_FONTSIZE = 14
_CBAR_LABEL = "Relative displacement (µm)"


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
