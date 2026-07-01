"""3-D flyer displacement animation for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..analysis.tilt import fit_plane
from ..io.alpss import load_probe_positions


def animate_pdv(data, positions_csv, output_path="pdv_animation.gif",
                 step=100, elev=20, azim=45, radius=750,
                 fps=15, dpi=100, focus_scale=2.0):
    """
    Produce a 3-D animation of flyer displacement from ``process_pdv`` output.

    Parameters
    ----------
    data : pd.DataFrame
        Output of ``process_pdv``.  Must contain a ``time`` column and at
        least one ``probe_<N>`` column (displacement in µm).
    positions_csv : str
        Path to the probe-position CSV (columns: ``probe_number``,
        ``x_position``, ``y_position``).
    output_path : str, optional
        File to save the animation.  Use ``.gif`` or ``.mp4``.
        Pass ``None`` to skip saving.
    step : int, optional
        Subsample every *step*-th row of ``data`` (default 100).
    elev : float, optional
        Elevation angle of the 3-D camera in degrees (default 20).
    azim : float, optional
        Azimuth angle of the 3-D camera in degrees (default 45).
        Together ``elev``/``azim`` set the viewing direction:
        ``elev=0, azim=0`` → pure side view along y-axis;
        ``elev=90`` → top-down view.
    radius : float, optional
        Radius of the circular plane patch in the same units as
        ``x_position``/``y_position`` (default 750 µm).
    fps : int, optional
        Frames per second for the saved animation (default 15).
    dpi : int, optional
        Resolution of the saved animation (default 100).
    focus_scale : float, optional
        Divisor applied to probe x/y positions to match the optical focus
        length used at acquisition (default 2.0, for the 125 mm focus setup).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    probe_locs = load_probe_positions(positions_csv, focus_scale=focus_scale)

    probe_cols = [c for c in data.columns if c.endswith("_pos")]
    probes = []
    for col in probe_cols:
        num = int(col.split("_")[1])
        loc = probe_locs[probe_locs["probe_number"] == num]
        if loc.empty:
            continue
        probes.append({
            "num": num,
            "col": col,
            "x":   float(loc["x_position"].values[0]),
            "y":   float(loc["y_position"].values[0]),
        })

    if not probes:
        raise ValueError("No probes matched between data columns and positions_csv.")

    time_ns = data["time"].values * 1e9          # seconds → ns

    # --- circular plane mesh ----------------------------------------------
    gg = np.linspace(-radius, radius, 50)
    XX, YY = np.meshgrid(gg, gg)
    circle_mask = XX ** 2 + YY ** 2 <= radius ** 2

    # --- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("displacement (µm)")

    z_all = np.concatenate([data[p["col"]].values for p in probes])
    z_min, z_max = np.nanmin(z_all), np.nanmax(z_all)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((1, 1, 1))

    probe_artists = []
    for p in probes:
        pt, = ax.plot([p["x"]], [p["y"]], [np.nan], "o", ms=8,
                      label=f"probe {p['num']}")
        probe_artists.append(pt)
    ax.legend(loc="upper left", fontsize=8)

    plane_surf = [None]   # list so closure can mutate it

    xs = np.array([p["x"] for p in probes])
    ys = np.array([p["y"] for p in probes])
    frames = list(range(0, len(data), step))

    def update(i):
        # update probe markers
        for artist, p in zip(probe_artists, probes):
            zval = data[p["col"]].iat[i]
            artist.set_data([p["x"]], [p["y"]])
            artist.set_3d_properties([zval])

        # remove previous plane
        if plane_surf[0] is not None:
            plane_surf[0].remove()
            plane_surf[0] = None

        # refit plane from probes with valid z at this step
        z_row = np.array([data[p["col"]].iat[i] for p in probes])
        fit = fit_plane(xs, ys, z_row)
        if not np.isnan(fit.a):
            ZZ = np.ma.array(fit.a * XX + fit.b * YY + fit.c, mask=~circle_mask)
            plane_surf[0] = ax.plot_surface(
                XX, YY, ZZ, color="red", alpha=0.3,
                edgecolor="k", linewidth=0.3, rstride=1, cstride=1)

        ax.set_title(f"t = {time_ns[i]:.1f} ns")
        return probe_artists

    anim = FuncAnimation(fig, update, frames=frames,
                         blit=False, interval=1000 / fps)

    if output_path:
        ext = output_path.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(output_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
        print(f"Saved -> {output_path}")

    plt.show()
    return anim
