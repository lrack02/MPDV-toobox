"""Combined velocity/tilt/3-D flyer animation for processed multi-probe PDV data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..analysis.tilt import fit_plane
from ..io.alpss import load_probe_positions


def animate_combined(data, positions_csv, output_path="combined.gif",
                     step=100, radius=750, rlim=None,
                     elev_side=0,  azim_side=0,
                     elev_iso=20,  azim_iso=45,
                     fps=15, dpi=100, focus_scale=2.0):
    """
    Four-panel time-synced animation:
      top-left     — velocity vs time (all probes, moving time cursor)
      top-right    — polar tilt (cumulative, coloured by time)
      bottom-left  — 3-D side view of flyer
      bottom-right — 3-D isometric view of flyer

    Parameters
    ----------
    data : pd.DataFrame
        Output of ``process_pdv``.
    positions_csv : str
        Path to the probe-position CSV.
    output_path : str, optional
        Save path (``.gif`` or ``.mp4``).  ``None`` skips saving.
    step : int, optional
        Subsample every *step*-th row (default 100).
    radius : float, optional
        Circular plane patch radius in µm (default 750).
    rlim : float, optional
        Fixed radial limit for polar plot in degrees.  Autoscales if ``None``.
    elev_side, azim_side : float, optional
        Camera angles for the side-view panel (default 0, 0).
    elev_iso, azim_iso : float, optional
        Camera angles for the isometric panel (default 20, 45).
    fps : int, optional
        Frames per second (default 15).
    dpi : int, optional
        Resolution of saved animation (default 100).
    focus_scale : float, optional
        Divisor applied to probe x/y positions to match the optical focus
        length used at acquisition (default 2.0, for the 125 mm focus setup).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    # ------------------------------------------------------------------
    # Probe positions
    # ------------------------------------------------------------------
    probe_locs = load_probe_positions(positions_csv, focus_scale=focus_scale)

    probe_cols = [c for c in data.columns if c.endswith("_pos")]
    probes = []
    for col in probe_cols:
        num = int(col.split("_")[1])
        loc = probe_locs[probe_locs["probe_number"] == num]
        if loc.empty:
            continue
        col_vel = col.replace("_pos", "_vel")
        probes.append({
            "num":     num,
            "col":     col,
            "col_vel": col_vel if col_vel in data.columns else None,
            "x": float(loc["x_position"].values[0]),
            "y": float(loc["y_position"].values[0]),
        })

    if not probes:
        raise ValueError("No probes matched between data columns and positions_csv.")

    xs = np.array([p["x"] for p in probes])
    ys = np.array([p["y"] for p in probes])

    # ------------------------------------------------------------------
    # Shared time / tilt arrays
    # ------------------------------------------------------------------
    frames     = list(range(0, len(data), step))
    time_ns    = data["time"].values * 1e9
    cmap       = plt.get_cmap("viridis")
    t_norm     = plt.Normalize(time_ns[frames[0]], time_ns[frames[-1]])
    theta_full = np.arctan2(data["ny"].values, data["nx"].values)
    r_full     = data["tilt_deg"].values

    # ------------------------------------------------------------------
    # Plane mesh (shared by both 3-D axes)
    # ------------------------------------------------------------------
    gg = np.linspace(-radius, radius, 50)
    XX, YY      = np.meshgrid(gg, gg)
    circle_mask = XX ** 2 + YY ** 2 <= radius ** 2

    z_all        = np.concatenate([data[p["col"]].values for p in probes])
    z_min, z_max = np.nanmin(z_all), np.nanmax(z_all)

    # ------------------------------------------------------------------
    # Figure layout  (2 × 2)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))

    ax_vel  = fig.add_subplot(2, 2, 1)                     # top-left
    ax_pol  = fig.add_subplot(2, 2, 2, projection="polar") # top-right
    ax_side = fig.add_subplot(2, 2, 3, projection="3d")    # bottom-left
    ax_iso  = fig.add_subplot(2, 2, 4, projection="3d")    # bottom-right

    # --- velocity plot: pre-draw full traces ---
    ax_vel.set_xlabel("Time (ns)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.set_title("Velocity vs time")
    for p in probes:
        if p["col_vel"]:
            ax_vel.plot(time_ns, data[p["col_vel"]].values,
                        lw=1, alpha=0.6, label=f"probe {p['num']}")
    ax_vel.legend(fontsize=7, loc="upper left")
    ax_vel.set_xlim(time_ns[frames[0]], time_ns[frames[-1]])
    vline = ax_vel.axvline(time_ns[frames[0]], color="red", lw=1.2)

    # --- polar axes ---
    ax_pol.set_rlabel_position(0)
    ax_pol.set_title("Polar tilt")
    ax_pol.grid(True)
    if rlim is not None:
        ax_pol.set_rlim(0, rlim)
    else:
        ax_pol.set_rlim(0, np.nanmax(r_full) * 1.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=t_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_pol, pad=0.12, label="Time (ns)", shrink=0.7)
    pol_line, = ax_pol.plot([], [], lw=1, color="k", zorder=1)
    pol_scat  = ax_pol.scatter([], [], c=[], cmap=cmap, norm=t_norm,
                               s=30, marker="*", zorder=2)
    pol_time  = ax_pol.text(0.05, 0.97, "", transform=ax_pol.transAxes,
                            fontsize=9, va="top")

    # --- helper: configure one 3-D axis ---
    def _setup_3d(ax, elev, azim, title):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (µm)", fontsize=7)
        ax.set_ylabel("y (µm)", fontsize=7)
        ax.set_zlabel("disp (µm)", fontsize=7)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect((1, 1, 1))   # cube box — z scale is exaggerated
        ax.set_title(title, fontsize=9)
        artists = []
        for p in probes:
            pt, = ax.plot([p["x"]], [p["y"]], [np.nan], "o", ms=6,
                          label=f"probe {p['num']}")
            artists.append(pt)
        ax.legend(loc="upper left", fontsize=6)
        return artists

    artists_side = _setup_3d(ax_side, elev_side, azim_side, "Side view")
    artists_iso  = _setup_3d(ax_iso,  elev_iso,  azim_iso,  "Isometric view")
    surf_side, surf_iso = [None], [None]

    # --- helper: update one 3-D axis ---
    def _update_3d(ax, probe_artists, surf, i):
        for artist, p in zip(probe_artists, probes):
            zval = data[p["col"]].iat[i]
            artist.set_data([p["x"]], [p["y"]])
            artist.set_3d_properties([zval])
        if surf[0] is not None:
            surf[0].remove()
            surf[0] = None
        z_row = np.array([data[p["col"]].iat[i] for p in probes])
        fit = fit_plane(xs, ys, z_row)
        if not np.isnan(fit.a):
            ZZ = np.ma.array(fit.a * XX + fit.b * YY + fit.c, mask=~circle_mask)
            surf[0] = ax.plot_surface(XX, YY, ZZ, color="red", alpha=0.3,
                                      edgecolor="k", linewidth=0.3,
                                      rstride=1, cstride=1)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    theta_acc, r_acc, t_acc = [], [], []

    def update(i):
        t = time_ns[i]

        # velocity cursor
        vline.set_xdata([t, t])

        # polar tilt
        if not np.isnan(r_full[i]):
            theta_acc.append(theta_full[i])
            r_acc.append(r_full[i])
            t_acc.append(t)
            pol_line.set_data(theta_acc, r_acc)
            pol_scat.set_offsets(np.column_stack([theta_acc, r_acc]))
            pol_scat.set_array(np.array(t_acc))
        pol_time.set_text(f"t = {t:.1f} ns")

        # both 3-D views
        _update_3d(ax_side, artists_side, surf_side, i)
        ax_side.set_title(f"Side view  t = {t:.1f} ns", fontsize=9)
        _update_3d(ax_iso,  artists_iso,  surf_iso,  i)
        ax_iso.set_title(f"Isometric  t = {t:.1f} ns", fontsize=9)

        return [vline, pol_line, pol_scat, pol_time] + artists_side + artists_iso

    anim = FuncAnimation(fig, update, frames=frames,
                         blit=False, interval=1000 / fps)
    plt.tight_layout()

    if output_path:
        ext = output_path.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(output_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
        print(f"Saved -> {output_path}")

    plt.show()
    return anim
