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

_LABEL_FONTSIZE = 14
_TICK_FONTSIZE = 12
_LEGEND_FONTSIZE = 11
_TITLE_FONTSIZE = 14
_MARKER_SIZE = 80
_LEGEND_MARKERSIZE = 9


def plot_radial_displacement(data, positions_csv, center_probe, legs, mask=None,
                              subsample=5000, focus_scale=2.0, ax=None):
    """Plot each probe's absolute displacement vs radial distance from the center probe.

    Each subsampled time step is drawn as one set of points, coloured by time,
    with a line connecting the center probe out through each leg's probes
    (sorted by radius). Marker shape and line style both encode which leg a
    probe sits on (or that it's the center probe), so the same physical leg
    can be tracked across radius as the flyer moves.

    center_probe : int
        Probe number at the center of the array (radial distance 0).
    legs : dict[float, list[int]]
        Maps leg angle in degrees (e.g. 0, 120, 240) to the probe numbers on
        that leg. Radial distance is computed from each probe's actual x/y
        position, not list order.
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

    if mask is None:
        mask = slice(None)
    t_masked = data["time"][mask].values * 1e9
    z_masked = {p["col"]: data[p["col"]][mask].values for p in probe_info}

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(t_masked.min(), t_masked.max())

    for i in range(0, len(t_masked), subsample):
        t = t_masked[i]
        color = cmap(norm(t))

        by_marker = {}
        for p in probe_info:
            z = z_masked[p["col"]][i]
            if np.isnan(z):
                continue
            group = by_marker.setdefault(p["marker"], {"r": [], "z": []})
            group["r"].append(p["r"])
            group["z"].append(z)
        for marker, vals in by_marker.items():
            ax.scatter(vals["r"], vals["z"], c=[t] * len(vals["r"]), cmap=cmap, norm=norm,
                       marker=marker, s=_MARKER_SIZE, alpha=0.75, edgecolors="none", zorder=2)

        # connecting line from the center probe out through each leg, one linestyle per leg
        z_center = z_masked[center_col][i] if center_col is not None else np.nan
        for leg in legs:
            points = []
            if not np.isnan(z_center):
                points.append((0.0, z_center))
            for p in probe_info:
                if p["leg"] != leg:
                    continue
                z = z_masked[p["col"]][i]
                if not np.isnan(z):
                    points.append((p["r"], z))
            if len(points) < 2:
                continue
            points.sort(key=lambda rz: rz[0])
            rs, zs = zip(*points)
            ax.plot(rs, zs, linestyle=leg_linestyles[leg], color=color,
                    linewidth=1.3, alpha=0.8, zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = ax.get_figure()
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Time (ns)", fontsize=_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)

    legend_handles = [Line2D([0], [0], marker=_CENTER_MARKER, color="k", linestyle="",
                              markersize=_LEGEND_MARKERSIZE, label=f"Center Probe")]
    for leg in sorted(legs):
        legend_handles.append(Line2D([0], [0], marker=leg_markers[leg], color="k",
                                      linestyle=leg_linestyles[leg], linewidth=1.3,
                                      markersize=_LEGEND_MARKERSIZE, label=f"{leg}° leg"))
    ax.legend(handles=legend_handles, loc="best", fontsize=_LEGEND_FONTSIZE, frameon=True)

    ax.set_xlabel(f"Radial Position From Center (µm)", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Displacement (µm)", fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE, direction="in", length=6, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # flyer travels downward, so larger displacement should read as "down" on the plot
    ax.invert_yaxis()

    fig.tight_layout()
    return ax
