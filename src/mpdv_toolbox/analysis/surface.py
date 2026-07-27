"""Flyer surface reconstruction from per-probe displacement at a given time."""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from ..io.alpss import load_probe_positions


def compute_relative_surface(data, positions_csv, t, reference_probe=None,
                              focus_scale=2.0, grid_size=300):
    """Reconstruct the flyer's relative surface height at time ``t`` from each
    probe's displacement, then interpolate onto a regular x/y grid.

    data : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.process.process_pdv`` (or any
        DataFrame with a ``time`` column and one ``probe_<N>_pos`` column per
        probe).
    positions_csv : str
        Path to a CSV with ``probe_number``, ``x_position``, ``y_position``.
    t : float
        Time (seconds, same units as ``data["time"]``) at which to plot. The
        nearest available row in ``data`` is used.
    reference_probe : int, optional
        Probe number to zero displacement against (dz = pos - pos_ref).
        Defaults to the mean displacement across all probes with valid data
        at this time.
    focus_scale : float, optional
        Passed to ``load_probe_positions`` (default 2.0).
    grid_size : int, optional
        Number of grid points per axis for the interpolated surface
        (default 300).

    Returns
    -------
    x, y, z : np.ndarray
        Raw probe x/y positions (um) and relative displacement (um) at ``t``,
        one value per probe with both a valid displacement and a known
        position.
    Xi, Yi, Zi : np.ndarray
        Cubic-interpolated grid of relative displacement (um), shape
        ``(grid_size, grid_size)``. ``Xi``/``Yi`` are the meshgrid coordinates.
    t_actual : float
        The actual matched time (seconds) used from ``data["time"]``.
    """
    positions = load_probe_positions(positions_csv, focus_scale=focus_scale)

    idx = int(np.argmin(np.abs(data["time"].values - t)))
    t_actual = data["time"].values[idx]

    probe_cols = {int(c.split("_")[1]): c for c in data.columns if c.endswith("_pos")}

    rows = []
    for probe_num, col in probe_cols.items():
        pos = data[col].values[idx]
        if not np.isnan(pos):
            rows.append({"probe_number": probe_num, "position_um": pos})

    if not rows:
        raise ValueError(f"No valid displacement data at t = {t_actual * 1e9:.1f} ns.")

    df = pd.DataFrame(rows)

    if reference_probe is not None:
        ref_rows = df.loc[df["probe_number"] == reference_probe, "position_um"]
        if ref_rows.empty:
            raise ValueError(f"Reference probe {reference_probe} has no valid "
                              f"displacement at t = {t_actual * 1e9:.1f} ns.")
        z_ref = ref_rows.values[0]
    else:
        z_ref = df["position_um"].mean()

    df["rel_position_um"] = df["position_um"] - z_ref

    merged = df.merge(positions, on="probe_number")
    if merged.empty:
        raise ValueError("No probes matched between data and positions_csv.")

    x = merged["x_position"].to_numpy()
    y = merged["y_position"].to_numpy()
    z = merged["rel_position_um"].to_numpy()

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

    return x, y, z, Xi, Yi, Zi, t_actual
