"""Loaders for ALPSS-processed multi-probe PDV output."""

import pandas as pd


def load_probe_positions(positions_csv, focus_scale=2.0):
    """Load probe x/y positions, scaled for the optical focus length used at acquisition.

    focus_scale=2.0 matches the 125 mm focus setup currently in use; pass 1.0 for
    unscaled positions.
    """
    probe_locs = pd.read_csv(positions_csv)
    probe_locs["probe_number"] = probe_locs["probe_number"].astype(int)
    probe_locs["x_position"] = probe_locs["x_position"] / focus_scale
    probe_locs["y_position"] = probe_locs["y_position"] / focus_scale
    return probe_locs


def load_shot_displacement(base):
    """Load a shot's combined ALPSS multipoint displacement CSV.

    ``base`` is the output file stem for one shot, e.g. ``{output_dir}/{PDV_FileName}``
    (matching ALPSS's ``alpss_multipoint_with_config`` output naming). Returns a
    DataFrame with a ``time`` column (s) and one ``probe_<N>`` column (m) per
    probe; probes absent from this shot are all-NaN.
    """
    return pd.read_csv(f"{base}-displacement.csv")


def load_shot_velocity(base):
    """Load a shot's combined ALPSS multipoint smoothed-velocity CSV.

    Same layout as ``load_shot_displacement`` but with velocity in m/s.
    """
    return pd.read_csv(f"{base}-velocity--smooth.csv")


def probe_columns(shot_df):
    """Probe numbers present as columns in a loaded shot DataFrame, in column order."""
    return [int(c.split("_")[1]) for c in shot_df.columns if c != "time"]


def extract_probe_series(shot_df, probe_num, scale=1.0):
    """Pull one probe's (time, value) series out of a combined shot DataFrame.

    Drops samples where that probe has no data (NaN) -- ALPSS pads a shot's
    shared time axis with NaN outside each probe's own valid window.
    """
    col = f"probe_{probe_num}"
    valid = shot_df[col].notna()
    return shot_df["time"][valid].to_numpy(), shot_df[col][valid].to_numpy() * scale
