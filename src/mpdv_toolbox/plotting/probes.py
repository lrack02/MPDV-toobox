"""Cross-probe start-time plots."""

import matplotlib.pyplot as plt
import numpy as np


def plot_start_time_vs_probe(start_times, reference_probe, ylim=(-100, 100), ax=None):
    """Scatter each probe's start time relative to ``reference_probe``, across all shots.

    start_times : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.start_times.collect_start_times``.
    """
    probe_cols = [c for c in start_times.columns if c.startswith("probe_")]
    probe_nums = [int(c.split("_")[1]) for c in probe_cols]
    ref_col = f"probe_{reference_probe}"

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for _, row in start_times.iterrows():
        ref = row.get(ref_col, np.nan)
        if np.isnan(ref):
            continue
        for col, num in zip(probe_cols, probe_nums):
            t = row[col]
            if not np.isnan(t):
                ax.scatter(num, (t - ref) * 1e9, color="steelblue", s=20)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Probe")
    ax.set_ylim(ylim)
    ax.set_ylabel(f"Start Time relative to Probe {reference_probe} (ns)")
    ax.set_title(f"Start Time vs Probe (relative to Probe {reference_probe})")
    ax.set_xticks(probe_nums)
    ax.get_figure().tight_layout()
    return ax


def plot_start_time_polar(start_times, probe_layout, reference_probe, ns_max=50, ax=None):
    """Polar plot of probe start-time delay relative to ``reference_probe``.

    probe_layout : dict[int, dict]
        Maps probe number to ``{'angle_deg': ..., 'ring': 'middle' | 'outer'}``,
        describing each probe's physical layout on the connector face.
    """
    ref_col = f"probe_{reference_probe}"
    r_middle, r_outer = 1.0, 2.4
    scale = 0.35 / ns_max

    probe_delays = {p: [] for p in probe_layout}
    for _, row in start_times.iterrows():
        ref = row.get(ref_col, np.nan)
        if np.isnan(ref):
            continue
        for p in probe_layout:
            t = row.get(f"probe_{p}", np.nan)
            if not np.isnan(t):
                delay_ns = (t - ref) * 1e9
                if abs(delay_ns) <= ns_max:
                    probe_delays[p].append(delay_ns)

    theta = np.linspace(0, 2 * np.pi, 500)

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for r_ref in (r_middle, r_outer):
        ax.plot(theta, [r_ref] * 500, color="k", ls="--", lw=1.1, alpha=0.55)
        for ns_off, ls, alpha in [(ns_max, ":", 0.45), (2 * ns_max, ":", 0.30)]:
            ax.plot(theta, [r_ref + ns_off * scale] * 500, color="gray", ls=ls, lw=0.8, alpha=alpha)
            ax.plot(theta, [r_ref - ns_off * scale] * 500, color="gray", ls=ls, lw=0.8, alpha=alpha)

    for r_ref in (r_middle, r_outer):
        for ns_off in (ns_max, 2 * ns_max):
            ax.annotate(f"+{ns_off} ns", xy=(np.deg2rad(5), r_ref + ns_off * scale),
                        fontsize=6, color="gray", va="center")
            ax.annotate(f"−{ns_off} ns", xy=(np.deg2rad(5), r_ref - ns_off * scale),
                        fontsize=6, color="gray", va="center")

    for p, info in probe_layout.items():
        ang = np.deg2rad(info["angle_deg"])
        r_ref = r_middle if info["ring"] == "middle" else r_outer
        delays = probe_delays[p]
        if not delays:
            continue
        r_vals = [r_ref + d * scale for d in delays]
        ax.scatter([ang] * len(r_vals), r_vals, color="steelblue", s=18, zorder=4)
        ax.annotate(f"P{p}", xy=(ang, r_ref), xytext=(ang, r_ref + 0.28),
                    ha="center", va="center", fontsize=9, fontweight="bold")

    ax.scatter([0], [0], color="steelblue", s=40, zorder=5)
    ax.annotate(f"P{reference_probe}", xy=(0, 0), xytext=(0, 0.18),
                ha="center", fontsize=9, fontweight="bold")

    ax.set_rticks([])
    ax.grid(False)
    ax.set_title(f"Start Time Delay relative to Probe {reference_probe}  (±{ns_max * 2} ns range)", pad=20)
    return ax
