"""Short-time Fourier transform matching the legacy scipy STFT behavior used by ALPSS."""

import numpy as np
from scipy.signal import ShortTimeFFT


def stft(voltage, fs, **inputs):
    """Calculate the STFT of a signal.

    ALPSS was originally built with a scipy STFT function that is now deprecated.
    This function replicates its behavior with the new ``ShortTimeFFT`` API,
    specifically how the time windows are calculated and how the boundaries are
    handled (no zero padding at the edges).
    """
    # calculate stft with the new scipy library function and zero padding the boundaries
    SFT = ShortTimeFFT.from_window(
        inputs["window"],
        fs=fs,
        nperseg=inputs["nperseg"],
        noverlap=inputs["noverlap"],
        mfft=inputs["nfft"],
        scale_to="magnitude",
        phase_shift=None,
    )
    Sx_full = SFT.stft(voltage, padding="zeros")
    t_full = SFT.t(len(voltage))
    f = SFT.f

    # calculate the time array for the legacy scipy stft function without zero padding on the boundaries
    t_legacy = np.arange(
        inputs["nperseg"] / 2,
        voltage.shape[-1] - inputs["nperseg"] / 2 + 1,
        inputs["nperseg"] - inputs["noverlap"],
    ) / float(fs)

    # find the time index in the new stft function that corresponds to where the legacy function time array begins
    t_idx = np.argmin(np.abs(t_full - t_legacy[0]))

    # crop the time array to the length of the legacy function
    t_crop = t_full[t_idx : t_idx + len(t_legacy)]

    # crop the stft magnitude array to the length of the legacy function
    Sx_crop = Sx_full[:, t_idx : t_idx + len(t_legacy)]

    return f, t_crop, Sx_crop
