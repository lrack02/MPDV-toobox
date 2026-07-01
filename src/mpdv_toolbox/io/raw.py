"""Loaders for raw single-probe oscilloscope PDV exports."""

import numpy as np


def load_oscilloscope_csv(filepath):
    """Load a raw oscilloscope CSV export (time, voltage columns, 30-line header).

    Returns the zeroed time array (s), voltage array, and the sample rate (Hz)
    inferred from the time column.
    """
    data = np.genfromtxt(filepath, delimiter=",", skip_header=30)

    time = data[:, 0]
    time = time - time[0]
    voltage = data[:, 1]

    fs = 1 / np.mean(np.diff(time))

    return time, voltage, fs
