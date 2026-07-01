"""Tilt/planarity plots for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import numpy as np


def plot_tilt_vs_time(data, mask=None, ax=None):
    if mask is None:
        mask = slice(None)
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(data["time"][mask] * 1e9, data["tilt_deg"][mask])
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("tilt (degrees)")
    return ax


def plot_planarity_vs_time(data, mask=None, ax=None):
    if mask is None:
        mask = slice(None)
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(data["time"][mask] * 1e9, data["r_squared"][mask])
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("r squared")
    ax.set_ylim([0, 1])
    return ax


def plot_polar_tilt(data, mask=None, ax=None):
    """Static polar plot of tilt direction/magnitude over time, coloured by time."""
    if mask is None:
        mask = data["tilt_deg"].notna()
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

    theta = np.arctan2(data["ny"][mask], data["nx"][mask])
    r = data["tilt_deg"][mask]

    ax.plot(theta, r, markersize=3, linewidth=1, color="k")
    cmap = plt.get_cmap("viridis")
    scatter = ax.scatter(theta, r, c=data["time"][mask] * 1e9, cmap=cmap, s=50, marker="*")

    cbar = ax.get_figure().colorbar(scatter, ax=ax)
    cbar.set_label("Time (ns)")

    ax.set_rlabel_position(0)
    ax.set_title("Polar Plot of Tilt")
    ax.grid(True)
    return ax
