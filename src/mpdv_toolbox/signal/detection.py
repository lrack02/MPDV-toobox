"""Event detection: mean-shift (CUSUM) and zero-crossing helpers."""

import numpy as np


def cusum(signal, mu0, sigma, h, k):
    """Detect a single mean shift from mu0 to mu1 using CUSUM.

    Returns:
    - Detection index
    - Estimated change point index
    - Full G[k] array
    - Score array s
    """
    # Score for general mean change
    Z = (signal - mu0) / (np.sqrt(sigma))
    s = -Z - k
    G = np.zeros_like(s)

    for k in range(1, len(s)):
        G[k] = max(G[k - 1] + s[k], 0)

        if G[k] > h:
            detect_idx = k
            S = np.cumsum(s[:detect_idx])
            change_idx = np.argmin(S)
            return detect_idx, change_idx, G, s

    # If no change detected
    return None, None, G, s


def find_zero_crossings(values):
    """Return indices where ``values`` changes sign.

    Useful for locating oscillation half-periods in a derivative trace
    (e.g. acceleration) around an impact event.
    """
    return np.where(np.diff(np.sign(values)))[0]


def find_impact_time(time, signal, baseline_window, min_shift=None, h=5.0, direction="auto"):
    """Detect the onset of a sustained step-change in ``signal`` (e.g. impact velocity change) using CUSUM.

    CUSUM is robust to sample-to-sample noise because it only declares a
    detection once enough cumulative evidence has built up -- a single noisy
    point can't trigger it -- and the returned time is the changepoint
    *before* the detection index, which corrects for CUSUM's inherent
    detection lag.

    Parameters
    ----------
    time, signal : array-like
        One probe's time (s) and signal (e.g. velocity, m/s) trace.
    baseline_window : (float, float)
        Time range assumed to be on the pre-impact plateau (after any launch
        transient has settled). Used to estimate the baseline mean/std that
        CUSUM measures deviations against, and marks the start of the search
        region (only ``time >= baseline_window[1]`` is searched).
    min_shift : float, optional
        Smallest change in ``signal`` that should count as an event (same
        units as ``signal``). Defaults to 3x the baseline noise std, so pure
        noise fluctuations don't trigger a false detection.
    h : float, optional
        CUSUM decision threshold in baseline-sigma units (default 5.0, a
        standard low-false-alarm-rate choice).
    direction : {"auto", "up", "down"}, optional
        Which way the event shifts ``signal``. "auto" (default) checks both
        directions and returns whichever fires first.

    Returns
    -------
    float or None
        Estimated event time, or None if no sustained shift was detected.
    """
    time = np.asarray(time)
    signal = np.asarray(signal)

    baseline = (time >= baseline_window[0]) & (time <= baseline_window[1])
    mu0 = signal[baseline].mean()
    std0 = signal[baseline].std()
    if std0 == 0:
        raise ValueError("Baseline window has zero variance; widen it or check the data.")

    if min_shift is None:
        min_shift = 3 * std0
    k = min_shift / (2 * std0)
    variance0 = std0 ** 2  # cusum()'s `sigma` argument is a variance, not a std

    search = time >= baseline_window[1]
    t_search = time[search]
    v_search = signal[search]

    candidates = []
    if direction in ("auto", "down"):
        _, change_idx, *_ = cusum(v_search, mu0, variance0, h, k)
        if change_idx is not None:
            candidates.append(t_search[change_idx])
    if direction in ("auto", "up"):
        _, change_idx, *_ = cusum(-v_search, -mu0, variance0, h, k)
        if change_idx is not None:
            candidates.append(t_search[change_idx])

    if not candidates:
        return None
    return min(candidates)
