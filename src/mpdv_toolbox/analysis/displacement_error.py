import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_analytical(voltage_df, time):
    """Extracts analytial signal for the time period where the alpss processor was conducted. Time may come from disp_df or other output"""
    t_start_idx = np.argwhere(time.values[0]==voltage_df["time"].values)[0][0]
    t_end_idx = np.argwhere(time.values[-1]==voltage_df["time"].values)[0][0]

    voltage = voltage_df.values[t_start_idx:t_end_idx+1,1:] #eliminate time column, end inclusive
    real = voltage[:, np.arange(0, voltage.shape[1]-1, step=2)] #real columns are even
    imag = voltage[:, np.arange(1, voltage.shape[1], step=2)] #imag columns are odd

    analytical = real + 1j * imag

    print(analytical.shape)
    return analytical

def signal_to_noise(analytical, noise_floor):
    """Estimate per-sample signal-to-noise ratio. 
    """
    amplitude = np.abs(analytical)

    return amplitude / noise_floor

def unwrap_phase(analytical):
    theta = np.angle(analytical)
    delta_theta = np.diff(theta, axis=0)
    delta_theta = np.insert(delta_theta, 0 , 1e-10, axis=0)
    phas = np.unwrap(theta)

    return theta, delta_theta, phas

def phas_to_pos(time, phas, lam, fit_time):
    mask = time < fit_time
    f0 = np.mean(np.gradient(phas[mask], np.mean(np.diff(time))))
    pos = lam / 2 / np.pi * (phas - f0 * time)

    return pos

def phas_uncert(delta_theta, noise_frac):
    var_theta = noise_frac
    a = var_theta[:-1,:]
    b = var_theta[1:,:]
    var_delta_theta = a + b
    var_delta_theta = np.insert(var_delta_theta, 0 , 1e-10, axis=0)
    
    p1 = 1 - norm.cdf(np.pi, loc=delta_theta, scale=np.sqrt(var_delta_theta))
    p2 = norm.cdf(-np.pi, loc=delta_theta, scale=np.sqrt(var_delta_theta))

    var_f = p2 * (1 - p2) + p1 * (1 - p1) + 2 * p1 * p2

    # this assumes independence between each time measurement. TODO: this is not necessarily true
    var_phas = var_theta + 4 * np.pi**2 * np.nancumsum(var_f, axis=0)

    return var_delta_theta, var_phas

def displacement_uncert(voltage_df, noise_frac_df, lam):
    time = noise_frac_df["time"]
    analytical = extract_analytical(voltage_df, time)
    snr_vals = noise_frac_df[noise_frac_df.columns[1:]].to_numpy()

    theta, delta_theta, phas = unwrap_phase(analytical)
    var_delta_theta, var_phas = phas_uncert(delta_theta, snr_vals)
    sigma_pos = lam / 2 * np.pi * np.sqrt(var_phas)

    sigma_disp_df = pd.DataFrame(data=np.column_stack((time, sigma_pos)), columns=noise_frac_df.columns)

    return sigma_disp_df


if __name__ == "__main__":
    from mpdv_toolbox.io.alpss import load_shot, filter_shot

    base = r"C:\Users\lucas\OneDrive - Johns Hopkins\Ramesh Lab - Research\Papers\MPDV\MPDV_velocity_experiments\2026-06-30_velocity_vacuum\output_data\C1--JHAMAA00004_2026-06-30_18-27-08_shot10--00000"

    voltage_df = load_shot(base, "voltage")
    disp_df = load_shot(base, "displacement")
    vel_df = load_shot(base, "velocity--smooth")
    sigma_vel_df = load_shot(base, "veluncert")
    noise_frac_df = load_shot(base, "noisefrac")

    column_names = disp_df.columns

    time = disp_df["time"].to_numpy()
    # mask = (time > 750e-9) & (time < 2200e-9)

    # # Filter to the time of interest
    # voltage_df = voltage_df[mask]
    # time = voltage_df["time"].to_numpy()

    analytical = extract_analytical(voltage_df, disp_df["time"])

    sigma_disp_df = displacement_uncert(voltage_df, noise_frac_df, 1550e-9)

    print(sigma_disp_df)

    plt.subplots(1,2,sharex=True)
    for probe in column_names[1:]:
        plt.subplot(121)
        plt.plot(disp_df["time"]*1e9, disp_df[probe]*1e6, label=probe)
        plt.fill_between(sigma_disp_df["time"]*1e9, 
                         (disp_df[probe] - sigma_disp_df[probe])*1e6, 
                         (disp_df[probe] + sigma_disp_df[probe])*1e6,
                         alpha=0.5)
        plt.xlabel("time (ns)")
        plt.ylabel("displacement (microns)")
        plt.subplot(122)
        plt.plot(vel_df["time"]*1e9, vel_df[probe]*1e6, label=probe)
        plt.fill_between(sigma_vel_df["time"]*1e9, 
                         (vel_df[probe] - sigma_vel_df[probe])*1e6, 
                         (vel_df[probe] + sigma_vel_df[probe])*1e6,
                         alpha=0.5)
        plt.xlabel("time (ns)")
        plt.ylabel("velocity (m/s)")
    plt.legend()
    plt.show()

    plt.subplots(1,2,sharex=True)
    for probe in column_names[1:]:
        plt.subplot(121)
        plt.plot(sigma_disp_df["time"]*1e9, sigma_disp_df[probe]*1e6, label=probe)
        plt.xlabel("time (ns)")
        plt.ylabel("displacement uncertainty (microns)")
        plt.subplot(122)
        plt.plot(noise_frac_df["time"]*1e9, 1/noise_frac_df[probe], label=probe)
        plt.xlabel("time (ns)")
        plt.ylabel("snr")
    plt.legend()
    plt.show()

    
    # dt = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    # time_idx = np.argmin(np.abs(1000e-9 - time))
    # pdf = norm.pdf(dt, loc=delta_theta[time_idx], scale=np.sqrt(1/snr[time_idx]))
    # plt.plot(dt, pdf)
    # # plt.xlim([-2 * np.pi, 2 * np.pi])
    # plt.show()



