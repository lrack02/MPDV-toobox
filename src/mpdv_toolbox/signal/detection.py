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
