"""Tilt/planarity plots for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mpdv_toolbox.analysis.tilt import tilt_vs_time


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


def plot_polar_tilt(popt_df, mask=None, ax=None, t_start = None):
    """Static polar plot of tilt direction/magnitude over time, coloured by time."""

    phi, theta = tilt_vs_time(popt_df = popt_df)

    if mask is None:
        mask = popt_df["B1"].notna()
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
    if t_start is None: 
        t_start = popt_df["time"][popt_df["B1"].notna()].values[0]


    ax.plot(theta[mask], phi[mask] * 180/np.pi, markersize=3, linewidth=1, color="k")

    cmap = plt.get_cmap("viridis")
    scatter = ax.scatter(theta[mask], phi[mask] * 180/np.pi, c=(popt_df["time"][mask] - t_start)* 1e9, cmap=cmap, s=50, marker="*")

    cbar = ax.get_figure().colorbar(scatter, ax=ax)
    cbar.set_label("Time (ns)")

    ax.set_rlabel_position(0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g°"))
    ax.set_title("Polar Plot of Tilt")
    ax.grid(True)
    return ax

if __name__ == "__main__":
    import pandas as pd
    popt_df = pd.read_csv(r"C:\Users\lucas\OneDrive - Johns Hopkins\Ramesh Lab - Research\Papers\MPDV\MPDV_velocity_experiments\2026-06-30_velocity_vacuum\output_data\C1--JHAMAA00004_2026-06-30_18-24-54_shot01--00000-fit_popt.csv")
    ax = plot_polar_tilt(popt_df = popt_df)
    plt.savefig("figures/tilt_vs_time2.png", dpi = 300)
    plt.show()

