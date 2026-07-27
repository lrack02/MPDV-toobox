"""Radial displacement-profile plots for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
                       marker=marker, s=_MARKER_SIZE, alpha=0.9, edgecolors="none", zorder=2)

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
