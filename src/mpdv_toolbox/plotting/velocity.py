"""Position/velocity vs time plots for processed multi-probe PDV data."""

import matplotlib.pyplot as plt


def plot_position_velocity(disp_df, vel_df, mask=None, figsize=(10, 5)):
    """Plot displacement and velocity vs time for every probe in ``data``.

    data : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.process.process_pdv``.
    mask : pd.Series[bool], optional
        Boolean mask into ``data`` rows to plot (e.g. a time window).
    """
    if mask is None:
        mask = slice(None)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    probe_nums = disp_df.columns[1:]
    for probe_num in probe_nums:
        label = probe_num
        axes[0].plot(disp_df["time"][mask] * 1e9, disp_df[label][mask], label=label)
        axes[1].plot(vel_df["time"][mask] * 1e9, vel_df[label][mask], label=label)

    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Position (microns)")
    axes[0].legend()

    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend()

    fig.tight_layout()
    return fig, axes
