"""Impact-time detection across all probes in processed multi-probe PDV data."""

import numpy as np
import pandas as pd
import os 
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from mpdv_toolbox.signal.detection import cusum
import matplotlib.pyplot as plt

# general function for a sinusoid
def sin_func(x, a, b, c, d):
    return a * np.sin(2 * np.pi * b * x + c) + d

def find_impact_time(time, voltage, launch_time = 0e9, fit_time = 200e-9, cen = 2e9, wid = 0.25e9, cusum_offset = 1, cusum_threshold = 1000):
    """Detect a probe's impact time from its velocity trace.

    time : numpy array
        Raw data of time from oscilloscope in seconds. Time shifted to match the processed pdv
    voltage : numpy array
        Raw volttage data from oscilloscope in volts. Time shifted to match the processed pdv
    launch_time : float
        Flyer launch time in seconds. Output found in results of processed PDV.
    fit_time : float
        User chosen length of sine fit to the carrier band in seconds.
    cen : float
        Frequency of the carrier band for a given probe. Output found in results of processed PDV
    wid : float
        User chosen half-width of the bandpass filter around cen, in Hz.

    Returns
    -------
    float
        impact time in seconds
    """
    # Apply narrow bandpass filter around the cen frequency to isolate carrier band
    freqs = fftfreq(time.size, np.mean(np.diff(time)))
    band_mask = (freqs > (cen - wid)) & (freqs < (cen + wid))
    voltage_fft = fft(voltage)
    voltage_filt_a = ifft(voltage_fft*band_mask)
    voltage_filt = voltage_filt_a.real

    plt.plot(time, np.abs(voltage_filt_a))
    plt.show()

    # Mask the signal to only the fitting time portion
    mask = (time > (launch_time + 50e-9)) & (time < (launch_time + fit_time))
    time_masked = time[mask]
    voltage_masked = voltage_filt[mask]

    # Seed initial guess for sine fit
    a0 = 1/2 * (voltage_masked.max() - voltage_masked.min())

    # Quick FFT-based seed for b0: use the strongest frequency actually
    # present within the bandpass window instead of assuming the carrier
    # sits exactly at cen. Just testing this out -- if it doesn't help,
    # fall back to the cen-based seed below.
    band_idx = np.where(band_mask)[0]
    b0 = freqs[band_idx[np.argmax(np.abs(voltage_fft[band_idx]))]]
    # b0 = cen  # previous seed, kept here in case the FFT-based estimate doesn't help

    # sin_func's phase term (2*pi*b*x + c) is referenced to x = 0, but the
    # analytic signal's phase is only known starting at time_masked[0]
    # (== launch_time), not t = 0. Extrapolate the phase back to t = 0 using
    # the b0 estimate before using it as the seed for c.
    phase_at_start = np.angle(voltage_filt_a[mask][0])
    phase_at_t0 = phase_at_start - 2 * np.pi * b0 * time_masked[0]
    c0_raw = phase_at_t0 + np.pi / 2
    c0 = np.mod(c0_raw + np.pi, 2 * np.pi) - np.pi  # wrap to (-pi, pi]
    d0 = 0

    # fit a sinusoid to the data
    popt, pcov = curve_fit(
        sin_func, time_masked, voltage_masked, p0=[a0, b0, c0, 0]
    )

    # create fit data for all time
    voltage_fit = sin_func(time, *popt)

    plt.plot(time_masked, voltage_masked)
    plt.plot(time_masked, sin_func(time_masked, *popt))
    plt.show()

    # subtract fit data from voltage_filt
    voltage_carrier_subtract = voltage_filt - voltage_fit

    plt.plot(time, voltage_carrier_subtract)
    plt.axvline(launch_time)
    plt.axvline(launch_time+fit_time)
    plt.show()

    # Impact Detection. Convert to analytical signal to get instantaneous magnitude
    mask_detection = time > launch_time
    time_detection = time[mask_detection]
    signal = -np.abs(hilbert(voltage_carrier_subtract)[mask_detection])

    sigma0 = np.var(signal[:500])  # cusum()'s `sigma` argument is a variance, not a std
    mu0 = np.mean(signal[:500])

    detect_idx, change_idx, G, s = cusum(signal = signal, mu0 = mu0, sigma = sigma0, h = cusum_threshold, k = cusum_offset)
    plt.plot(time_detection, s)
    plt.show()

    # change_idx is CUSUM's changepoint estimate, correcting for the
    # detection lag inherent to detect_idx (the threshold-crossing index).
    return time_detection[change_idx]

    

def extract_data(pdv_filepath, sample_rate = 40e9, header_lines = 1, time_to_skip = 0.1e-6, time_to_take = 1500e-9):
    """
    time_to_skip : float
        Time to skip for processing data. This is the ALPSS parameter used to process velocity data.
        It is required in order to time match to time on processed data.
    
    """

    t_step = 1 / sample_rate
    rows_to_skip = header_lines + time_to_skip / t_step
    nrows = time_to_take / t_step

    data = pd.read_csv(
        pdv_filepath,
        header=None,
        skiprows=int(rows_to_skip),
        nrows=int(nrows),
    )

    time = data.iloc[:, 0].to_numpy()
    time = time - time[0]
    voltage = data.iloc[:, 1].to_numpy()
    return time, voltage

def impact_times(results_df, inputs_df, multipoint_meta_data, input_data_path, shot_number, fit_time, cusum_offset, cusum_threshold):
    impact_times_df = pd.DataFrame(columns = [
        "Probe Number",
        "Impact Time"
    ])

    for idx, probe_number in enumerate(results_df["probe_number"].values):
        mask = (multipoint_meta_data["Exp_ID"] == shot_number) & (multipoint_meta_data["Probe_Number"] == probe_number)
        multipoint_metadata_masked = multipoint_meta_data[mask]

        # Get PDV filename
        pdv_filename = multipoint_metadata_masked["PDV_FileName"].values[0]
        pdv_filepath = os.path.join(input_data_path, pdv_filename) + ".csv"

        # Get processing parameters. Use .iloc since idx is a positional
        # counter and results_df/inputs_df may not have a clean 0..N-1 index.
        cen = results_df["Carrier Frequency"].iloc[idx]
        launch_time = results_df["Signal Start Time"].iloc[idx]
        sample_rate = inputs_df["sample_rate"].iloc[idx]
        wid = inputs_df["wid"].iloc[idx]
        header_lines = inputs_df["header_lines"].iloc[idx]
        time_to_skip = inputs_df["time_to_skip"].iloc[idx]
        time_to_take = inputs_df["time_to_take"].iloc[idx]

        time, voltage = extract_data(pdv_filepath, sample_rate, header_lines, time_to_skip, time_to_take)

        impact_time = find_impact_time(time, voltage, launch_time, fit_time, cen, 0.25e9, cusum_offset, cusum_threshold)

        new_row = {
            "Probe Number": probe_number,
            "Impact Time": impact_time
        }

        impact_times_df = pd.concat([impact_times_df, pd.DataFrame([new_row])], ignore_index=True)

    return impact_times_df


def velocity_at_impact(vel_df, t_eval):
    t_eval_idx = np.argmin(np.abs(vel_df["time"] - t_eval))

    v_f = np.mean(vel_df[vel_df.columns[1:]].values[t_eval_idx, :])
    v_f_std = np.std(vel_df[vel_df.columns[1:]].values[t_eval_idx, :])

    return v_f, v_f_std
