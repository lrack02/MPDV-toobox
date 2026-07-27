"""Radial displacement-profile plots for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..io.alpss import load_probe_positions

_LEG_MARKERS = ["^", "s", "D", "v", "P", "X", "h"]
_CENTER_MARKER = "o"
_LEG_LINESTYLES = [
    "--",                 # dashed
    ":",                  # dotted
    "-.",                 # dash-dot
    (0, (3, 1, 1, 1)),    # dash-dot-dot
    (0, (5, 1)),          # long dash
    (0, (1, 1)),          # densely dotted
    (0, (3, 5, 1, 5)),    # loosely dash-dot
]
_LINE_COLOR = "0.35"  # neutral gray -- color is reserved for velocity

_LABEL_FONTSIZE = 14
_TICK_FONTSIZE = 12
_LEGEND_FONTSIZE = 11
_TITLE_FONTSIZE = 14
_MARKER_SIZE = 80
_LEGEND_MARKERSIZE = 9


def _nearest_index(time_arr, t):
    idx = np.searchsorted(time_arr, t)
    if idx <= 0:
        return 0
    if idx >= len(time_arr):
        return len(time_arr) - 1
    before = idx - 1
    return idx if abs(time_arr[idx] - t) < abs(time_arr[before] - t) else before


def plot_radial_displacement(data, positions_csv, center_probe, legs, timestep,
                              focus_scale=2.0, ax=None):
    """Plot each probe's absolute displacement vs radial distance from the center probe,
    at a series of evenly-spaced points in time spanning the full ``data["time"]`` range.
    Each time step draws a line connecting the center probe out through each leg's
    probes (sorted by radius). Marker shape and line style encode which leg a probe
    sits on (or that it's the center probe); point color encodes that probe's
    velocity at its time step, on a color scale shared across all time steps.

    center_probe : int
        Probe number at the center of the array (radial distance 0).
    legs : dict[float, list[int]]
        Maps leg angle in degrees (e.g. 0, 120, 240) to the probe numbers on
        that leg. Radial distance is computed from each probe's actual x/y
        position, not list order.
    timestep : float
        Spacing (seconds, same units as ``data["time"]``) between plotted time
        steps. An array of time steps at this spacing is built across the
        full range of ``data["time"]``; the nearest available row in ``data``
        is used for each one (steps that map to the same row are only drawn
        once).
    """
    pos = load_probe_positions(positions_csv, focus_scale=focus_scale)

    def _xy(probe_num):
        row = pos.loc[pos["probe_number"] == probe_num]
        return float(row["x_position"].values[0]), float(row["y_position"].values[0])

    x_c, y_c = _xy(center_probe)
    leg_markers = {leg: _LEG_MARKERS[i % len(_LEG_MARKERS)] for i, leg in enumerate(sorted(legs))}
    leg_linestyles = {leg: _LEG_LINESTYLES[i % len(_LEG_LINESTYLES)] for i, leg in enumerate(sorted(legs))}

    probe_cols = {int(c.split("_")[1]): c for c in data.columns if c.endswith("_pos")}

    center_col = probe_cols.get(center_probe)
    probe_info = []
    if center_col is not None:
        probe_info.append({"num": center_probe, "col": center_col,
                           "r": 0.0, "marker": _CENTER_MARKER, "leg": None})
    for leg, probe_nums in legs.items():
        marker = leg_markers[leg]
        for num in probe_nums:
            col = probe_cols.get(num)
            if col is None:
                continue
            x, y = _xy(num)
            r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
            probe_info.append({"num": num, "col": col, "r": r, "marker": marker, "leg": leg})

    time_arr = data["time"].values
    t_min, t_max = time_arr.min(), time_arr.max()
    requested = np.arange(t_min, t_max + timestep / 2, timestep)
    indices = sorted({_nearest_index(time_arr, t) for t in requested})

    def _vel(probe_num, idx):
        col = f"probe_{probe_num}_vel"
        return data[col].values[idx] if col in data.columns else np.nan

    vel_vals = [v for idx in indices for v in (_vel(p["num"], idx) for p in probe_info)
                if not np.isnan(v)]
    if not vel_vals:
        raise ValueError(f"No valid velocity data between t = {t_min * 1e9:.1f} "
                          f"and {t_max * 1e9:.1f} ns.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(vel_vals), max(vel_vals))

    for idx in indices:
        by_marker = {}
        for p in probe_info:
            z = data[p["col"]].values[idx]
            if np.isnan(z):
                continue
            v = _vel(p["num"], idx)
            color = cmap(norm(v)) if not np.isnan(v) else "0.7"
            group = by_marker.setdefault(p["marker"], {"r": [], "z": [], "colors": []})
            group["r"].append(p["r"])
            group["z"].append(z)
            group["colors"].append(color)
        for marker, vals in by_marker.items():
            ax.scatter(vals["r"], vals["z"], color=vals["colors"],
                       marker=marker, s=_MARKER_SIZE, alpha=0.9,
                       edgecolors="k", linewidths=0.7, zorder=2)

        # connecting line from the center probe out through each leg
        z_center = data[center_col].values[idx] if center_col is not None else np.nan
        for leg in legs:
            points = []
            if not np.isnan(z_center):
                points.append((0.0, z_center))
            for p in probe_info:
                if p["leg"] != leg:
                    continue
                z = data[p["col"]].values[idx]
                if not np.isnan(z):
                    points.append((p["r"], z))
            if len(points) < 2:
                continue
            points.sort(key=lambda rz: rz[0])
            rs, zs = zip(*points)
            ax.plot(rs, zs, linestyle=leg_linestyles[leg], color=_LINE_COLOR,
                    linewidth=1.3, alpha=0.8, zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Velocity (m/s)", fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    legend_handles = [Line2D([0], [0], marker=_CENTER_MARKER, color=_LINE_COLOR, linestyle="",
                              markersize=_LEGEND_MARKERSIZE, label="Center Probe")]
    for leg in sorted(legs):
        legend_handles.append(Line2D([0], [0], marker=leg_markers[leg], color=_LINE_COLOR,
                                      linestyle=leg_linestyles[leg], linewidth=1.3,
                                      markersize=_LEGEND_MARKERSIZE, label=f"{leg}° leg"))
    ax.legend(handles=legend_handles, loc="best", fontsize=_LEGEND_FONTSIZE, frameon=True)

    ax.set_xlabel("Radial Position From Center (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Displacement (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_title(f"t = {t_min * 1e9:.0f}-{t_max * 1e9:.0f} ns  (Δt = {timestep * 1e9:.1f} ns)",
                 fontsize=_TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE, direction="in", length=6, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # flyer travels downward, so larger displacement should read as "down" on the plot
    ax.invert_yaxis()

    fig.tight_layout()
    return fig, ax


def plot_radial_displacement_3d(data, positions_csv, center_probe, legs, timestep,
                                 focus_scale=2.0, elev=30, azim=-60, ax=None):
    """3-D version of ``plot_radial_displacement``: each probe is plotted at its real
    x/y position and its actual displacement (not projected to radial distance), at a
    series of evenly-spaced points in time spanning the full ``data["time"]`` range,
    all overlaid in one 3-D axes. Each time step draws a line connecting the center
    probe out through each leg's probes (sorted by radius). Marker shape and line
    style encode which leg a probe sits on (or that it's the center probe); point
    color encodes that probe's displacement relative to the center probe (deviation
    from center) at its time step, on a diverging scale shared across all time steps.

    center_probe : int
        Probe number at the center of the array.
    legs : dict[float, list[int]]
        Maps leg angle in degrees (e.g. 0, 120, 240) to the probe numbers on
        that leg.
    timestep : float
        Spacing (seconds, same units as ``data["time"]``) between plotted time
        steps -- see ``plot_radial_displacement``.
    elev, azim : float, optional
        3-D camera angles in degrees (defaults 30, -60, matplotlib's own
        default view) -- the legs can overlap heavily at some angles, so it's
        worth adjusting these per data set. Note: matplotlib's 3-D z-axis
        label can occasionally fail to render at some elev/azim + data-extent
        combinations (a known mplot3d quirk, not specific to this data) --
        if the z-axis label goes missing after changing these, try a
        different angle.
    """
    pos = load_probe_positions(positions_csv, focus_scale=focus_scale)

    def _xy(probe_num):
        row = pos.loc[pos["probe_number"] == probe_num]
        return float(row["x_position"].values[0]), float(row["y_position"].values[0])

    x_c, y_c = _xy(center_probe)
    leg_markers = {leg: _LEG_MARKERS[i % len(_LEG_MARKERS)] for i, leg in enumerate(sorted(legs))}
    leg_linestyles = {leg: _LEG_LINESTYLES[i % len(_LEG_LINESTYLES)] for i, leg in enumerate(sorted(legs))}

    probe_cols = {int(c.split("_")[1]): c for c in data.columns if c.endswith("_pos")}

    center_col = probe_cols.get(center_probe)
    probe_info = []
    if center_col is not None:
        probe_info.append({"num": center_probe, "col": center_col, "x": x_c, "y": y_c,
                           "r": 0.0, "marker": _CENTER_MARKER, "leg": None})
    for leg, probe_nums in legs.items():
        marker = leg_markers[leg]
        for num in probe_nums:
            col = probe_cols.get(num)
            if col is None:
                continue
            x, y = _xy(num)
            r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
            probe_info.append({"num": num, "col": col, "x": x, "y": y,
                               "r": r, "marker": marker, "leg": leg})

    time_arr = data["time"].values
    t_min, t_max = time_arr.min(), time_arr.max()
    requested = np.arange(t_min, t_max + timestep / 2, timestep)
    indices = sorted({_nearest_index(time_arr, t) for t in requested})

    def _deviation(p, idx, z_center):
        z = data[p["col"]].values[idx]
        if np.isnan(z) or np.isnan(z_center):
            return np.nan
        return z - z_center

    dev_vals = []
    for idx in indices:
        z_center = data[center_col].values[idx] if center_col is not None else np.nan
        dev_vals.extend(d for p in probe_info
                        if not np.isnan(d := _deviation(p, idx, z_center)))
    if not dev_vals:
        raise ValueError(f"No valid displacement data between t = {t_min * 1e9:.1f} "
                          f"and {t_max * 1e9:.1f} ns.")

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    cmap = plt.get_cmap("RdBu_r")
    clim = max(abs(min(dev_vals)), abs(max(dev_vals)))
    norm = plt.Normalize(-clim, clim)

    for idx in indices:
        z_center = data[center_col].values[idx] if center_col is not None else np.nan

        by_marker = {}
        for p in probe_info:
            z = data[p["col"]].values[idx]
            if np.isnan(z):
                continue
            dev = _deviation(p, idx, z_center)
            color = cmap(norm(dev)) if not np.isnan(dev) else "0.7"
            group = by_marker.setdefault(p["marker"], {"x": [], "y": [], "z": [], "colors": []})
            group["x"].append(p["x"])
            group["y"].append(p["y"])
            group["z"].append(z)
            group["colors"].append(color)
        for marker, vals in by_marker.items():
            ax.scatter(vals["x"], vals["y"], vals["z"], color=vals["colors"],
                       marker=marker, s=_MARKER_SIZE, alpha=0.9, depthshade=False,
                       edgecolors="k", linewidths=0.7, zorder=2)

        # connecting line from the center probe out through each leg
        if center_col is None or np.isnan(z_center):
            continue
        for leg in legs:
            leg_points = []
            for p in probe_info:
                if p["leg"] != leg:
                    continue
                z = data[p["col"]].values[idx]
                if not np.isnan(z):
                    leg_points.append((p["r"], p["x"], p["y"], z))
            if not leg_points:
                continue
            leg_points.sort(key=lambda v: v[0])
            xs, ys, zs = zip(*[(x_c, y_c, z_center)] + [(x, y, z) for _, x, y, z in leg_points])
            ax.plot(xs, ys, zs, linestyle=leg_linestyles[leg], color=_LINE_COLOR,
                    linewidth=1.3, alpha=0.8, zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.18)
    cbar.set_label("Deviation from Center Probe (µm)", fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    legend_handles = [Line2D([0], [0], marker=_CENTER_MARKER, color=_LINE_COLOR, linestyle="",
                              markersize=_LEGEND_MARKERSIZE, label="Center Probe")]
    for leg in sorted(legs):
        legend_handles.append(Line2D([0], [0], marker=leg_markers[leg], color=_LINE_COLOR,
                                      linestyle=leg_linestyles[leg], linewidth=1.3,
                                      markersize=_LEGEND_MARKERSIZE, label=f"{leg}° leg"))
    # ax.legend(handles=legend_handles, loc="best", fontsize=_LEGEND_FONTSIZE, frameon=True)

    ax.set_xlabel("x position (µm)", fontsize=_LABEL_FONTSIZE, labelpad=14)
    ax.set_ylabel("y position (µm)", fontsize=_LABEL_FONTSIZE, labelpad=14)
    ax.set_zlabel("Displacement (µm)", fontsize=_LABEL_FONTSIZE, labelpad=14)
    ax.set_title(f"t = {t_min * 1e9:.0f}-{t_max * 1e9:.0f} ns  (Δt = {timestep * 1e9:.1f} ns)",
                 fontsize=_TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE, pad=6)
    ax.view_init(elev=elev, azim=azim)

    # flyer travels downward, so larger displacement should read as "down" on the plot
    ax.invert_zaxis()

    # matplotlib's tight_layout/bbox_inches="tight" don't reliably account for 3-D
    # axis labels, so the z-label can get clipped -- reserve explicit margins instead.
    fig.subplots_adjust(left=0.15, right=0.78, top=0.92, bottom=0.2)
    return fig, ax


def plot_probe_layout(positions_csv, center_probe, legs, focus_scale=2.0, ax=None):
    """Reference diagram of the probe array's real geometry.

    Draws each probe at its actual x/y position (from ``positions_csv``),
    with a line connecting the center probe out through each leg's probes
    (sorted by radial distance from the center, same convention as
    ``plot_radial_displacement``). Marker shape and line style encode which
    leg a probe sits on. Useful as a quick visual check that a ``legs`` dict
    matches the real physical layout.

    positions_csv : str
        Path to a CSV with ``probe_number``, ``x_position``, ``y_position``.
    center_probe : int
        Probe number at the center of the array.
    legs : dict[float, list[int]]
        Maps leg angle in degrees to the probe numbers on that leg.
    focus_scale : float, optional
        Passed to ``load_probe_positions`` (default 2.0).
    """
    pos = load_probe_positions(positions_csv, focus_scale=focus_scale)

    def _xy(probe_num):
        row = pos.loc[pos["probe_number"] == probe_num]
        return float(row["x_position"].values[0]), float(row["y_position"].values[0])

    x_c, y_c = _xy(center_probe)
    leg_markers = {leg: _LEG_MARKERS[i % len(_LEG_MARKERS)] for i, leg in enumerate(sorted(legs))}
    leg_linestyles = {leg: _LEG_LINESTYLES[i % len(_LEG_LINESTYLES)] for i, leg in enumerate(sorted(legs))}

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    def _label(num, x, y):
        ax.annotate(str(num), (x, y), textcoords="offset points", xytext=(8, 8),
                    fontsize=_TICK_FONTSIZE, fontweight="bold")

    ax.scatter([x_c], [y_c], marker=_CENTER_MARKER, s=_MARKER_SIZE, color="white",
               edgecolors="k", linewidths=1.0, zorder=3)
    _label(center_probe, x_c, y_c)

    for leg, probe_nums in legs.items():
        marker = leg_markers[leg]
        points = [(0.0, x_c, y_c)]
        for num in probe_nums:
            x, y = _xy(num)
            r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
            points.append((r, x, y))
            ax.scatter([x], [y], marker=marker, s=_MARKER_SIZE, color="white",
                       edgecolors="k", linewidths=1.0, zorder=3)
            _label(num, x, y)
        points.sort(key=lambda v: v[0])
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, linestyle=leg_linestyles[leg], color=_LINE_COLOR,
                linewidth=1.3, zorder=1)

    legend_handles = [Line2D([0], [0], marker=_CENTER_MARKER, color=_LINE_COLOR, linestyle="",
                              markerfacecolor="white", markeredgecolor="k",
                              markersize=_LEGEND_MARKERSIZE, label="Center Probe")]
    for leg in sorted(legs):
        legend_handles.append(Line2D([0], [0], marker=leg_markers[leg], color=_LINE_COLOR,
                                      linestyle=leg_linestyles[leg], linewidth=1.3,
                                      markerfacecolor="white", markeredgecolor="k",
                                      markersize=_LEGEND_MARKERSIZE, label=f"{leg}° leg"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=_LEGEND_FONTSIZE, frameon=True)

    ax.set_xlabel("x position (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("y position (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_title("Probe Array Layout", fontsize=_TITLE_FONTSIZE)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE, direction="in", length=6, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    return fig, ax
