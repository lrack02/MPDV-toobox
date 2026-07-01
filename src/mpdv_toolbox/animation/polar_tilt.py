"""Polar tilt-trajectory animation for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


def animate_polar_tilt(data, output_path="polar_tilt.gif",
                       step=100, rlim=None, fps=15, dpi=100):
    """
    Animate the polar tilt plot from ``process_pdv`` output.

    The polar angle is the azimuthal direction of the tilt (arctan2(ny, nx))
    and the radial distance is the tilt magnitude in degrees.  Each frame
    draws the cumulative path up to that point, coloured by time.

    Parameters
    ----------
    data : pd.DataFrame
        Output of ``process_pdv``.  Must contain ``time``, ``nx``, ``ny``,
        ``tilt_deg`` columns.
    output_path : str, optional
        File to save the animation.  Use ``.gif`` or ``.mp4``.
        Pass ``None`` to skip saving.
    step : int, optional
        Subsample every *step*-th valid row (default 100).
    rlim : float, optional
        Maximum radial limit in degrees.  Defaults to the maximum tilt value.
    fps : int, optional
        Frames per second (default 15).
    dpi : int, optional
        Resolution of saved animation (default 100).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    valid = data["tilt_deg"].notna()
    df = data[valid].iloc[::step].reset_index(drop=True)

    theta_vals = np.arctan2(df["ny"].values, df["nx"].values)
    r_vals     = df["tilt_deg"].values
    t_vals     = df["time"].values * 1e9

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(t_vals.min(), t_vals.max())

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_rlabel_position(0)
    ax.set_title("Polar tilt animation")
    ax.grid(True)
    if rlim is not None:
        ax.set_rlim(0, rlim)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1, label="Time (ns)")

    line, = ax.plot([], [], linewidth=1, color="k", zorder=1)
    scat  = ax.scatter([], [], c=[], cmap=cmap, norm=norm,
                       s=40, marker="*", zorder=2)
    time_text = ax.text(0.05, 0.97, "", transform=ax.transAxes,
                        fontsize=10, verticalalignment="top")

    def init():
        line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        time_text.set_text("")
        return line, scat, time_text

    def update(i):
        line.set_data(theta_vals[:i+1], r_vals[:i+1])
        scat.set_offsets(np.column_stack([theta_vals[:i+1], r_vals[:i+1]]))
        scat.set_array(t_vals[:i+1])
        time_text.set_text(f"t = {t_vals[i]:.1f} ns")
        return line, scat, time_text

    anim = FuncAnimation(fig, update, frames=len(t_vals),
                         init_func=init, interval=1000 / fps, blit=True)

    if output_path:
        ext = output_path.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(output_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
        print(f"Saved -> {output_path}")

    plt.show()
    return anim
