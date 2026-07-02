"""Position/velocity vs time plots for processed multi-probe PDV data."""

import matplotlib.pyplot as plt


def plot_position_velocity(data, mask=None, figsize=(10, 5)):
    """Plot displacement and velocity vs time for every probe in ``data``.

    data : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.process.process_pdv``.
    mask : pd.Series[bool], optional
        Boolean mask into ``data`` rows to plot (e.g. a time window).
    """
    if mask is None:
        mask = slice(None)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    probe_nums = [c.split("_")[1] for c in data.columns if c.endswith("_pos")]
    for probe_num in probe_nums:
        label = f"probe_{probe_num}"
        axes[0].plot(data["time"][mask] * 1e9, data[f"{label}_pos"][mask], label=label)
        axes[1].plot(data["time"][mask] * 1e9, data[f"{label}_vel"][mask], label=label)

    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Position (microns)")
    axes[0].legend()

    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend()

    fig.tight_layout()
    return fig, axes
