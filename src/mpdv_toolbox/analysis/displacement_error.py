import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm


def signal_to_noise(real, imag, noise_floor):
    """Estimate per-sample signal-to-noise ratio. 
    """
    amplitude = np.hypot(real, imag)

    return amplitude / noise_floor

def unwrap_phase(real, imag):
    analytical = real + 1j * imag
    theta = np.angle(analytical)
    delta_theta = np.diff(theta)
    delta_theta = np.insert(delta_theta, 0 , 1e-10)
    phas = np.unwrap(theta)

    return theta, delta_theta, phas

def phas_to_pos(time, phas, lam, fit_time):
    mask = time < fit_time
    f0 = np.mean(np.gradient(phas[mask], np.mean(np.diff(time))))
    pos = lam / 2 / np.pi * (phas - f0 * time)

    return pos

def phas_uncert(delta_theta, snr):
    var_theta = 1/snr
    a = var_theta[:-1]
    b = var_theta[1:]
    var_delta_theta = a + b
    var_delta_theta = np.insert(var_delta_theta, 0 , 1e-10)
    
    p1 = 1 - norm.cdf(np.pi, loc=delta_theta, scale=np.sqrt(var_delta_theta))
    p2 = norm.cdf(-np.pi, loc=delta_theta, scale=np.sqrt(var_delta_theta))

    var_f = p2 * (1 - p2) + p1 * (1 - p1) + 2 * p1 * p2

    # this assumes independence between each time measurement. TODO: this is not necessarily true
    var_phas = var_theta + 4 * np.pi**2 * np.cumsum(var_f)

    return var_delta_theta, var_phas

def displacement_uncert(real, imag, snr):
    theta, delta_theta, phas = unwrap_phase(real, imag)
    var_delta_theta, var_phas = phas_uncert(delta_theta, snr)
    sigma_pos = lam / 2 * np.pi * np.sqrt(var_phas)

    return sigma_pos




if __name__ == "__main__":
    import pandas as pd

    fname = r"C:\Users\lucas\OneDrive - Johns Hopkins\Ramesh Lab - Research\Papers\MPDV\MPDV_velocity_experiments\2026-06-30_velocity_vacuum\output_data\C1--JHAMAA00004_2026-06-30_18-31-54_shot31--00000-voltage.csv"
    probe_num = 19

    df = pd.read_csv(fname)
    valid = df[f"probe_{probe_num}_real"].notna()
    time = df["time"][valid].to_numpy()

    mask = (time > 750e-9) & (time < 2200e-9)

    time = time[mask]
    real = df[f"probe_{probe_num}_real"][valid].to_numpy()[mask]
    imag = df[f"probe_{probe_num}_imag"][valid].to_numpy()[mask]

    noise_floor = 1e-4  # TODO: incorporate alpss noise frac
    fs = 1.0 / (time[1] - time[0])
    lam = 1.55e-6  # from this shot's ALPSS -inputs.csv

    uncertainty_multiplier = 100  # exaggerate the uncertainty band so it's visible against the full displacement

    theta, delta_theta, phas = unwrap_phase(real, imag)

    snr = signal_to_noise(real, imag, noise_floor)

    var_delta_theta, var_phas = phas_uncert(delta_theta, snr)

    sigma_pos = lam / 2 * np.pi * np.sqrt(var_phas)

    plt.plot(time*1e9, sigma_pos*1e6)
    plt.xlabel("Time (ns)")
    plt.ylabel("Displacement Uncertainty (microns)")
    plt.savefig("displacement_uncert2.png",dpi=300)
    plt.show()

    idx = 10000
    
    dt = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    time_idx = np.argmin(np.abs(1000e-9 - time))
    pdf = norm.pdf(dt, loc=delta_theta[time_idx], scale=np.sqrt(1/snr[time_idx]))
    plt.plot(dt, pdf)
    # plt.xlim([-2 * np.pi, 2 * np.pi])
    plt.show()

    plt.plot()


