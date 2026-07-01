"""Spectrogram plots for single-probe oscilloscope PDV data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..io.raw import load_oscilloscope_csv
from ..signal.stft import stft


def plot_spectrogram(filepath, xlim=[500, 3000], ylim=[0, 10]):
    time, voltage, fs = load_oscilloscope_csv(filepath)

    nperseg = 512
    noverlap = 435
    nfft = 5120

    # calculate the short time fourier transform
    f, t, Zxx = stft(voltage, fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # calculate magnitude of Zxx
    mag = np.abs(Zxx)

    plt.imshow(
        10 * np.log10(mag**2),
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[time[0] / 1e-9, time[-1] / 1e-9, f[0] / 1e9, f[-1] / 1e9],
        cmap="viridis",
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Frequency (GHz)")
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_spectrogram_multi(directory, pattern="*", xlim=[500, 3000], ylim=[0, 10], figsize=(10, 5), savepath=None):
    files = sorted(Path(directory).glob(pattern))

    plt.subplots(1, len(files), figsize=figsize)
    for i, file in enumerate(files):
        print(file)
        plt.subplot(1, len(files), i + 1)
        plot_spectrogram(file, xlim=xlim, ylim=ylim)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()
