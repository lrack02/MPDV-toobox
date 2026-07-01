from pathlib import Path

import numpy as np
from scipy.signal import ShortTimeFFT
import matplotlib.pyplot as plt

def stft(voltage, fs, **inputs):
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

def plot_spectrogram(filepath, xlim = [500, 3000], ylim = [0, 10]):
    data = np.genfromtxt(filepath, delimiter=",", skip_header=30)

    # put the data into numpy arrays. Zero the time data
    time = data[:,0]
    time = time - time[0]
    voltage = data[:,1]

    # calculate the sample rate from the experimental data
    fs = 1 / np.mean(np.diff(time))


    nperseg = 512
    noverlap= 435
    nfft = 5120 

    # calculate the short time fourier transform
    f, t, Zxx = stft(voltage, fs, window = "hann", nperseg = nperseg, noverlap = noverlap, nfft = nfft)

    # calculate magnitude of Zxx
    mag = np.abs(Zxx)

    plt.imshow(
        10 * np.log10(mag**2),
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[time[0] / 1e-9, time[-1] / 1e-9, f[0] / 1e9, f[-1] / 1e9],
        cmap='viridis',
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
