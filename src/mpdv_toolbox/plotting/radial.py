"""Radial displacement-profile plots for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import numpy as np

from ..io.alpss import load_probe_positions


def plot_radial_displacement(data, positions_csv, reference_probe, mask=None,
                              subsample=5000, focus_scale=2.0, ax=None):
    """Plot each probe's displacement, relative to ``reference_probe``, vs radial distance from it.

    Each subsampled time step is drawn as one set of points, coloured by time.
    """
    pos = load_probe_positions(positions_csv, focus_scale=focus_scale)

    ref_loc = pos.loc[pos["probe_number"] == reference_probe]
    x_ref = ref_loc["x_position"].values[0]
    y_ref = ref_loc["y_position"].values[0]

    probe_cols = [c for c in data.columns if c.endswith("_pos")]
    probe_info = []
    for col in probe_cols:
        num = int(col.split("_")[1])
        row = pos.loc[pos["probe_number"] == num]
        if row.empty:
            continue
        x, y = row["x_position"].values[0], row["y_position"].values[0]
        r = np.sqrt((x - x_ref) ** 2 + (y - y_ref) ** 2)
        probe_info.append({"num": num, "col": col, "r": r})

    if mask is None:
        mask = slice(None)
    t_masked = data["time"][mask].values * 1e9
    z_masked = {p["col"]: data[p["col"]][mask].values for p in probe_info}
    z_ref = z_masked[f"probe_{reference_probe}_pos"]

    if ax is None:
        _, ax = plt.subplots()

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(t_masked.min(), t_masked.max())

    for i in range(0, len(t_masked), subsample):
        t = t_masked[i]
        rs, dzs = [], []
        for p in probe_info:
            z = z_masked[p["col"]][i]
            z_center = z_ref[i]
            if np.isnan(z) or np.isnan(z_center):
                continue
            rs.append(p["r"])
            dzs.append(z - z_center)
        if rs:
            ax.scatter(rs, dzs, c=[t] * len(rs), cmap=cmap, norm=norm, s=25, alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = ax.get_figure()
    fig.colorbar(sm, ax=ax, label="Time (ns)")

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Radial distance from probe {reference_probe} (µm)")
    ax.set_ylabel(f"Δ Displacement from probe {reference_probe} (µm)")
    ax.set_title("Radial displacement difference")
    fig.tight_layout()
    return ax
