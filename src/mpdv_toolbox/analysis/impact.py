"""Impact-time detection across all probes in processed multi-probe PDV data."""

import numpy as np
import pandas as pd

from ..signal.detection import find_impact_time


def find_impact_times(data, baseline_window, min_shift=None, h=5.0, direction="auto"):
    """Detect each probe's impact time from its velocity trace.

    data : pd.DataFrame
        Output of ``mpdv_toolbox.analysis.process.process_pdv`` (or any
        DataFrame with a ``time`` column and one ``probe_<N>_vel`` column
        per probe).
    baseline_window : (float, float)
        Time range (s) assumed to be on the pre-impact velocity plateau for
        every probe -- see ``mpdv_toolbox.signal.detection.find_impact_time``.
    min_shift, h, direction :
        Passed through to ``find_impact_time`` for every probe.

    Returns
    -------
    pd.DataFrame
        Columns: ``probe_number``, ``impact_time`` (seconds, NaN if undetected).
    """
    probe_cols = [c for c in data.columns if c.endswith("_vel")]

    rows = []
    for col in probe_cols:
        probe_num = int(col.split("_")[1])
        valid = data[col].notna()
        t_impact = find_impact_time(
            data["time"][valid].to_numpy(),
            data[col][valid].to_numpy(),
            baseline_window=baseline_window,
            min_shift=min_shift,
            h=h,
            direction=direction,
        )
        rows.append({"probe_number": probe_num,
                     "impact_time": t_impact if t_impact is not None else np.nan})

    return pd.DataFrame(rows).sort_values("probe_number").reset_index(drop=True)
