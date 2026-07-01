"""Cross-shot probe start-time analysis from processed PDV displacement data."""

import glob
import os

import numpy as np
import pandas as pd


def find_start_time(time, displacement, threshold=1e-6):
    """Return the first time at which ``|displacement|`` exceeds ``threshold``.

    ``time``/``displacement`` are same-length Series in consistent units
    (threshold defaults to meters, matching raw ALPSS displacement output).
    """
    valid = displacement.dropna()
    moving = valid[valid.abs() > threshold]
    if moving.empty:
        return np.nan
    return float(time.loc[moving.index[0]])


def collect_start_times(data_dir, pattern="*--combined.csv", threshold=1e-6):
    """Find per-probe start times across every processed shot CSV in ``data_dir``.

    Returns a DataFrame with one row per file and one column per ``probe_*``
    column found across all files.
    """
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))

    rows = []
    all_probes = set()
    for f in files:
        df = pd.read_csv(f)
        probe_cols = [c for c in df.columns if c.startswith("probe_")]
        all_probes.update(probe_cols)
        row = {"filename": os.path.basename(f)}
        for col in probe_cols:
            row[col] = find_start_time(df["time"], df[col], threshold=threshold)
        rows.append(row)

    all_probes = sorted(all_probes, key=lambda x: int(x.split("_")[1]))
    return pd.DataFrame(rows, columns=["filename"] + all_probes)
