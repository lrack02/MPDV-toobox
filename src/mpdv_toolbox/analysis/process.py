"""Aggregate multi-probe ALPSS output into a single time-aligned DataFrame with tilt/planarity."""

import numpy as np
import pandas as pd

from ..io.alpss import (
    extract_probe_series,
    load_probe_positions,
    load_shot_displacement,
    load_shot_velocity,
    probe_columns,
)
from .tilt import fit_plane, fit_plane_batch


def process_pdv(base, positions_csv, z_thresh=20.0, output_csv=None):
    """
    Process one shot's ALPSS multipoint output into a single aligned DataFrame.

    Reads the shot's combined ``-displacement.csv``/``-velocity--smooth.csv``
    (one column per probe, produced by ALPSS's multipoint mode -- see
    ``alpss_multipoint_with_config``), resamples every probe onto a common
    time grid, then computes a plane fit across probe positions at every
    time step.

    Parameters
    ----------
    base : str
        ALPSS multipoint output file stem for one shot, e.g.
        r"...\\output_data\\C1--JHAMAA00004_2026-06-30_18-24-54_shot01--00000"
        (i.e. ``{output_dir}/{PDV_FileName}``, with no ``-displacement.csv``/
        ``-velocity--smooth.csv`` suffix).
    positions_csv : str
        Path to a CSV with columns ``x_position``, ``y_position``,
        ``probe_number`` (e.g. ``Spatial_Distribution/2026-02-18/positions.csv``).
    z_thresh : float, optional
        Displacement threshold in microns used to zero each probe's time axis.
        Time is shifted so t=0 is when the probe's displacement first crosses
        this value.  Defaults to 20.0 µm.
    output_csv : str, optional
        If given, the result DataFrame is saved to this path.

    Returns
    -------
    pd.DataFrame
        Column order:

        * ``time``  — seconds
        * ``probe_<N>``  — displacement in microns, one column per probe
        * ``nx``, ``ny``, ``nz``  — unit normal of fitted plane
        * ``tilt_deg``  — tilt angle from z-axis in degrees
        * ``r_squared`` — goodness of fit of the plane (unreliable when flyer is flat)
        * ``fit_rmse``  — RMS residual of the plane fit in µm (lower = better, flat-safe)
    """
    probe_locs = load_probe_positions(positions_csv, focus_scale=2.0)  # TEMPORARY FOR 125 mm focus

    disp_df = load_shot_displacement(base)
    try:
        vel_df = load_shot_velocity(base)
    except Exception:
        vel_df = None

    # ------------------------------------------------------------------
    # Load displacement/velocity data for every probe present in the shot
    # ------------------------------------------------------------------
    probes = []
    for probe_num in probe_columns(disp_df):
        try:
            tvals, zvals = extract_probe_series(disp_df, probe_num, scale=1e6)
            if len(tvals) == 0:
                print(f"Warning: probe {probe_num} has no displacement data; skipping.")
                continue

            loc = probe_locs[probe_locs["probe_number"] == probe_num]
            if loc.empty:
                print(f"Warning: probe {probe_num} not found in positions file; "
                      "setting x/y to NaN (excluded from plane fit).")
                x, y = np.nan, np.nan
            else:
                x = float(loc["x_position"].values[0])
                y = float(loc["y_position"].values[0])

            t_zero = tvals[np.argmin(np.abs(zvals - z_thresh))]
            tvals = tvals - t_zero

            if vel_df is not None:
                tv_vel, vvals = extract_probe_series(vel_df, probe_num, scale=1.0)
                tv_vel = tv_vel - t_zero
            else:
                tv_vel, vvals = None, None

            probes.append({"num": probe_num, "x": x, "y": y,
                           "t": tvals, "z": zvals,
                           "t_vel": tv_vel, "v": vvals,
                           "t0": t_zero})
        except Exception as e:
            print(f"Warning: could not load probe {probe_num}: {e}")

    if not probes:
        raise ValueError("No probe data could be loaded.")

    # ------------------------------------------------------------------
    # Build common time grid and interpolate every probe onto it
    # ------------------------------------------------------------------
    all_times = np.hstack([p["t"] for p in probes])
    t0, t1 = all_times.min(), all_times.max()
    # Use a fixed grid spacing to avoid near-zero diffs from offset per-probe grids
    dt = 1 / 128e9
    common_t = np.arange(t0, t1 + dt / 2, dt)
    Nt = len(common_t)

    df_dict = {"time": common_t}
    for p in probes:
        z_interp = np.interp(common_t, p["t"], p["z"], left=np.nan, right=np.nan)
        df_dict[f'probe_{p["num"]}_pos'] = z_interp
        p["z_interp"] = z_interp

        if p["t_vel"] is not None:
            v_interp = np.interp(common_t, p["t_vel"], p["v"], left=np.nan, right=np.nan)
        else:
            v_interp = np.full(Nt, np.nan)
        df_dict[f'probe_{p["num"]}_vel'] = v_interp

    # ------------------------------------------------------------------
    # Plane fit at every time step.
    #
    # For speed, probes with known (non-NaN) x/y that have data across
    # the full window are handled with a single batched fit. Time steps
    # where any probe value is NaN fall back to a per-step loop.
    # ------------------------------------------------------------------
    xs = np.array([p["x"] for p in probes])
    ys = np.array([p["y"] for p in probes])
    z_matrix = np.column_stack([p["z_interp"] for p in probes])  # (Nt, Nprobes)

    nx_arr = np.full(Nt, np.nan)
    ny_arr = np.full(Nt, np.nan)
    nz_arr = np.full(Nt, np.nan)
    tilt_arr = np.full(Nt, np.nan)
    r2_arr = np.full(Nt, np.nan)
    rmse_arr = np.full(Nt, np.nan)

    xy_valid = ~(np.isnan(xs) | np.isnan(ys))

    # --- fast path: all position-valid probes have finite z at this step ---
    all_finite = ~np.any(np.isnan(z_matrix[:, xy_valid]), axis=1)  # (Nt,)
    if all_finite.any() and xy_valid.sum() >= 3:
        xv = xs[xy_valid]
        yv = ys[xy_valid]
        z_sub = z_matrix[np.ix_(all_finite, np.where(xy_valid)[0])]  # (N, Nprobes)

        _, _, _, nx, ny, nz, tilt_deg, r2, rmse = fit_plane_batch(xv, yv, z_sub)

        nx_arr[all_finite] = nx
        ny_arr[all_finite] = ny
        nz_arr[all_finite] = nz
        tilt_arr[all_finite] = tilt_deg
        r2_arr[all_finite] = r2
        rmse_arr[all_finite] = rmse

    # --- slow path: per-step loop for time steps with some NaNs ---
    partial = ~all_finite   # steps not handled above
    for i in np.where(partial)[0]:
        z_row = z_matrix[i]
        valid = xy_valid & ~np.isnan(z_row)
        if valid.sum() < 3:
            continue
        fit = fit_plane(xs[valid], ys[valid], z_row[valid])
        nx_arr[i] = fit.nx
        ny_arr[i] = fit.ny
        nz_arr[i] = fit.nz
        tilt_arr[i] = fit.tilt_deg
        r2_arr[i] = fit.r_squared
        rmse_arr[i] = fit.rmse

    df_dict["nx"] = nx_arr
    df_dict["ny"] = ny_arr
    df_dict["nz"] = nz_arr
    df_dict["tilt_deg"] = tilt_arr
    df_dict["r_squared"] = r2_arr
    df_dict["fit_rmse"] = rmse_arr

    result = pd.DataFrame(df_dict)

    t0_df = pd.DataFrame({
        "probe_number": [p["num"] for p in probes],
        "t0":           [p["t0"] for p in probes],
    })

    if output_csv:
        result.to_csv(output_csv, index=False)
        t0_csv = output_csv.replace(".csv", "_t0.csv")
        t0_df.to_csv(t0_csv, index=False)
        print(f"Saved -> {output_csv}")
        print(f"Saved -> {t0_csv}")

    return result, t0_df
