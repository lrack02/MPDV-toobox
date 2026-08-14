"""Loaders for ALPSS-processed multi-probe PDV output."""

import pandas as pd
import numpy as np
from mpdv_toolbox.analysis.displacement_error import displacement_uncert
import re
from pathlib import Path


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


def load_shot(base, shot_type = "displacement", delays_df=None):
    """Load a shot's combined ALPSS multipoint displacement CSV.

    base: The output file stem for one shot, e.g. ``{output_dir}/{PDV_FileName}``
    (matching ALPSS's ``alpss_multipoint_with_config`` output naming). Returns a
    DataFrame with a ``time`` column (s) and one ``probe_<N>`` column (m) per
    probe; probes absent from this shot are all-NaN.

    delays_df: Optional relative probe delay dataframe with ``probe_number`` and
    ``delay`` (seconds) columns. Each probe's data is shifted so that
    ``new_time = time - delay``, then resampled back onto the original
    (unshifted) time grid. Probes present in the shot but missing from
    ``delays_df`` are left unshifted (delay=0). If ``delays_df`` is omitted,
    no shifting is applied and the raw CSV is returned as-is.
    """
    data = pd.read_csv(f"{base}-{shot_type}.csv")

    if delays_df is None:
        return data

    time = data["time"].to_numpy()
    delay_map = dict(zip(delays_df["probe_number"], delays_df["delay"]))

    data_df = pd.DataFrame({"time": time})
    for col in data.columns:
        if col == "time":
            continue
        probe_number = int(col.split("_")[1])
        delay = delay_map.get(probe_number, 0.0)
        data_df[col] = np.interp(time + delay, time, data[col].to_numpy(), left=np.nan, right=np.nan)

    return data_df

def list_bases(output_dir):
    """Unique shot base filenames (the `base` load_shot()/save_displacement_uncert()
    expect) in an ALPSS multipoint output_data directory, e.g.
    C1--JHAMAA00004_2026-06-30_18-24-54_shot01--00000
    """
    output_dir = Path(output_dir)
    pattern = re.compile(r"^(.*--\d+)(?:-.*|_probe\d+-plots\.png)$")

    bases = set()
    for f in output_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            bases.add(m.group(1))
    return sorted(bases)

def filter_shot(data_df, uncert_df, threshold=10e-6):
    mask = uncert_df.values[:, 1:] < threshold
    data_df_filtered = data_df.copy()
    data_df_filtered.iloc[:, 1:] = np.where(mask, data_df.values[:, 1:], np.nan)
    return data_df_filtered