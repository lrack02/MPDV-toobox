"""Flyer surface reconstruction plots (heatmap / 3-D) at a given point in time."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from mpdv_toolbox.io.alpss import load_probe_positions
from mpdv_toolbox.analysis.surface_fit import surface
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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


def probe_coverage_radius(positions_csv, disp_df, center_probe="probe_10",
                          focus_scale=1.45):
    """Distance in meters from the center probe to the farthest measured probe."""
    probe_locs = load_probe_positions(positions_csv, focus_scale=focus_scale)
    probe_locs["probe_label"] = "probe_" + probe_locs["probe_number"].astype(str)
    if center_probe not in probe_locs["probe_label"].values:
        raise ValueError(f"{center_probe} is missing from the probe positions")
    center = probe_locs.loc[probe_locs["probe_label"] == center_probe].iloc[0]
    measured = [label for label in probe_locs["probe_label"]
                if label in disp_df.columns and disp_df[label].notna().any()]
    if not measured:
        raise ValueError("No measured probes match the probe positions")
    probe_locs = probe_locs[probe_locs["probe_label"].isin(measured)]
    return np.hypot(probe_locs["x_position"] - center["x_position"],
                    probe_locs["y_position"] - center["y_position"]).max() / 1e6


def flyer_surface_plot_limits(disp_df, popt, times, positions_csv,
                              reference_probe="probe_10", rrange=None,
                              focus_scale=1.45):
    """Shared z span and deflection color scale for a set of 3-D snapshots.

    Each snapshot's z span includes its fitted surface and measured points.
    The largest span is returned in microns, together with a color scale that
    covers deflections at every requested time.
    """
    if rrange is None:
        rrange = probe_coverage_radius(positions_csv, disp_df, reference_probe,
                                       focus_scale)
    probe_locs = load_probe_positions(positions_csv, focus_scale=focus_scale)
    probe_labels = [f"probe_{number}" for number in probe_locs["probe_number"]
                    if f"probe_{number}" in disp_df.columns]
    radius = np.linspace(0, rrange, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    rv, thetav = np.meshgrid(radius, theta)
    xy = np.column_stack((rv.ravel() * np.cos(thetav.ravel()),
                          rv.ravel() * np.sin(thetav.ravel())))

    z_spans = []
    w_values_all = []
    time = disp_df["time"].to_numpy()

    def finite_limits(arrays):
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("No finite surface or probe values at the requested times")
        return values.min(), values.max()

    for t_eval in times:
        idx = np.argmin(np.abs(t_eval - time))
        fitted_z = surface(xy, *popt.iloc[idx, 1:].to_numpy()) * 1e6
        measured_z = disp_df.iloc[idx][probe_labels].to_numpy(dtype=float) * 1e6
        reference_z = disp_df[reference_probe].iloc[idx] * 1e6
        z_min, z_max = finite_limits((fitted_z, measured_z))
        z_spans.append(z_max - z_min)
        w_values_all.extend((fitted_z - reference_z, measured_z - reference_z))

    return max(z_spans), plt.Normalize(*finite_limits(w_values_all))


def plot_flyer_surface_3d(disp_df, popt, t_eval, positions_csv, reference_probe="probe_10", rrange = None,  zrange = None,
                           focus_scale=1.45, grid_size=300, elev=20, azim=45, ax=None, norm=None):
    """3-D surface reconstruction of the flyer at time ``t_eval``.

    Pass shared ``zrange`` (microns) and ``norm`` to keep a series of plots on
    the same z-axis span and color scale.
    """

    time = disp_df["time"]
    t_eval_idx = np.argmin(np.abs(t_eval - time))
    p = tuple(popt.values[t_eval_idx,1:])
    if rrange is None:
        rrange = probe_coverage_radius(positions_csv, disp_df, reference_probe,
                                       focus_scale)

    # Generate meshgrid of flyer surface
    r = np.linspace(0,rrange, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    rv, thetav = np.meshgrid(r, theta)

    # Convert to cartesian
    xv, yv = rv * np.cos(thetav), rv * np.sin(thetav)

    # Reshape and put into surface model
    x = np.vstack((np.reshape(xv, xv.size), np.reshape(yv, xv.size))).T
    z = surface(x, *p)
    zv = np.reshape(z, (int(np.sqrt(z.size)),int(np.sqrt(z.size))))

    # Gather actual data
    probe_locs = load_probe_positions(positions_csv, focus_scale=focus_scale)

    # Keep only probes that are actually present in disp_df, in probe_locs order
    probe_locs["probe_label"] = "probe_" + probe_locs["probe_number"].astype(str)
    missing_probes = probe_locs.loc[~probe_locs["probe_label"].isin(disp_df.columns), "probe_label"].tolist()
    if missing_probes:
        print(f"Warning: {missing_probes} not found in disp_df, excluding from surface fit")
    probe_locs = probe_locs[probe_locs["probe_label"].isin(disp_df.columns)].reset_index(drop=True)

    x_values = probe_locs['x_position'].values / 1e6 #convert from microns to meters
    y_values = probe_locs['y_position'].values / 1e6 #convert from microns to meters
    x_data = np.vstack((x_values, y_values)).T

    # Order the probes
    disp_df_ordered = pd.DataFrame([])
    for probe_num in probe_locs["probe_number"].values:
        probe_label = "probe_" + str(probe_num)
        disp_df_ordered[probe_label] = disp_df[probe_label]

    # gather z data at correct time
    z_values = disp_df_ordered.values[t_eval_idx, :]

    # gather deflection data (w)
    w_values = z_values - disp_df[reference_probe].values[t_eval_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize = (10,10), subplot_kw={"projection": "3d"})
    else:
        fig = ax.get_figure()

    # Color both the fitted surface and measured points by deflection relative
    # to the same probe, using one range that includes both datasets.
    reference_z = disp_df[reference_probe].values[t_eval_idx]
    surface_w_um = (zv - reference_z) * 1e6
    probe_w_um = w_values * 1e6
    color_values = np.concatenate((surface_w_um.ravel(), probe_w_um))
    color_values = color_values[np.isfinite(color_values)]
    if color_values.size == 0:
        raise ValueError("No finite surface or probe deflections at the requested time")
    if norm is None:
        norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
    cmap = plt.get_cmap("coolwarm")

    ax.plot_surface(xv*1e6, yv*1e6, -zv*1e6,
                    facecolors=cmap(norm(surface_w_um)), shade=False,
                    linewidth=0, antialiased=False, alpha=0.25)

    # Plot the actual data with the same color normalization as the fit.
    P = ax.scatter3D(x_values*1e6, y_values*1e6, -z_values*1e6, s=50,
                     c=probe_w_um, cmap=cmap, norm=norm, edgecolor='k')

    fig.colorbar(P, ax=ax, label="deflection w (microns)", shrink=0.75, pad=0.15)

    ax.set_xlabel("x (microns)")
    ax.set_ylabel("y (microns)")
    ax.set_zlabel("z (microns)")
    z_at_time = np.concatenate((zv.ravel(), z_values)) * 1e6
    z_at_time = z_at_time[np.isfinite(z_at_time)]
    if z_at_time.size == 0:
        raise ValueError("No finite surface or probe heights at the requested time")
    if zrange is None:
        zrange = z_at_time.max() - z_at_time.min()
    z_center = (z_at_time.min() + z_at_time.max()) / 2
    # Draw z downward while keeping Matplotlib's lower x/y edges at the near
    # corner. Show the original displacement values on the z tick labels.
    ax.set_zlim(-(z_center + zrange / 2), -(z_center - zrange / 2))
    ax.zaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{0 if abs(value) < 1e-10 else -value:g}")
    )
    return fig, ax