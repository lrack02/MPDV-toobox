"""Impact-detection plots for processed multi-probe PDV velocity data."""

import matplotlib.pyplot as plt
import numpy as np


def plot_impact_detection(data, impact_times, mask=None, ax=None):
    """Overlay each probe's detected impact time on its velocity trace.

    data : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.process.process_pdv``.
    impact_times : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.impact.find_impact_times``
        (columns: ``probe_number``, ``impact_time``).
    mask : pd.Series[bool], optional
        Boolean mask into ``data`` rows to plot (e.g. a time window).
    """
    if mask is None:
        mask = slice(None)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for _, row in impact_times.iterrows():
        probe_num = int(row["probe_number"])
        col = f"probe_{probe_num}_vel"
        if col not in data.columns:
            continue
        line, = ax.plot(data["time"][mask] * 1e9, data[col][mask], label=f"probe_{probe_num}")
        if not np.isnan(row["impact_time"]):
            ax.axvline(row["impact_time"] * 1e9, color=line.get_color(), linestyle="--", alpha=0.6)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()
    return ax
