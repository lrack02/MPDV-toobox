"""Loaders for ALPSS-processed multi-probe PDV output."""

import numpy as np
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


def load_probe_displacement(fname, ch_name, probe_num):
    """Load one probe's ALPSS displacement CSV.

    Returns time (s) and displacement (um).
    """
    path = f"{fname}{ch_name}_probe{probe_num}-displacement.csv"
    raw = np.genfromtxt(path, delimiter=",")
    return raw[:, 0], raw[:, 1] * 1e6


def load_probe_velocity(fname, ch_name, probe_num):
    """Load one probe's ALPSS smoothed velocity CSV.

    Returns time (s) and velocity (m/s).
    """
    path = f"{fname}{ch_name}_probe{probe_num}-velocity--smooth.csv"
    raw = np.genfromtxt(path, delimiter=",")
    return raw[:, 0], raw[:, 1]
