import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def signal_to_noise(real, imag, noise_floor):
    """Estimate per-sample signal-to-noise ratio. 
    """
    amplitude = np.hypot(real, imag)

    return amplitude / noise_floor

def unwrap_phase(real, imag):
    analytical = real + 1j * imag
    a = analytical[:-1]
    b = analytical[1:]
    delta_theta = np.angle(b/a)
    delta_theta = np.insert(delta_theta,0,0)
    phas = np.cumsum(delta_theta)

    return phas

def phas_to_pos(time, phas, lam, fit_time):
    mask = time < fit_time
    f0 = np.mean(np.gradient(phas[mask], np.mean(np.diff(time))))
    pos = lam / 2 / np.pi * (phas - f0 * time)

    return pos

def phas_uncert(snr):
    var_theta = 1/snr
    a = var_theta[:-1]
    b = var_theta[1:]
    var_delta_theta = a + b
    var_delta_theta = np.insert(var_delta_theta, 0 , 0)

    var_phas = np.cumsum(var_delta_theta)

    sigma_phas = np.sqrt(var_phas)

    return sigma_phas

if __name__ == "__main__":
    import pandas as pd

    fname = r"C:\Users\lucas\OneDrive - Johns Hopkins\Ramesh Lab - Research\Papers\MPDV\MPDV_velocity_experiments\2026-06-30_velocity_vacuum\output_data\C1--JHAMAA00004_2026-06-30_18-31-54_shot31--00000-voltage.csv"
    probe_num = 19

    df = pd.read_csv(fname)
    valid = df[f"probe_{probe_num}_real"].notna()
    time = df["time"][valid].to_numpy()
    real = df[f"probe_{probe_num}_real"][valid].to_numpy()
    imag = df[f"probe_{probe_num}_imag"][valid].to_numpy()

    noise_floor = 1e-3  # TODO: incorporate alpss noise frac
    fs = 1.0 / (time[1] - time[0])
    lam = 1.55e-6  # from this shot's ALPSS -inputs.csv

    uncertainty_multiplier = 100  # exaggerate the uncertainty band so it's visible against the full displacement

    phas = unwrap_phase(real, imag)

    pos = phas_to_pos(time, phas, lam, 500e-9)

    snr = signal_to_noise(real, imag, noise_floor)

    sigma_phas = phas_uncert(snr)

    sigma_pos = lam / 2 / np.pi * sigma_phas

    mask = (time > 750e-9) & (time < 2200e-9)
    plt.plot(time[mask]*1e9, sigma_pos[mask]*1e6 - sigma_pos[mask][0]*1e6)
    plt.xlabel("Time (ns)")
    plt.ylabel("Displacement Uncertainty (microns)")
    plt.savefig("displacement_uncert.png",dpi=300)
    plt.show()

    plt.plot(time[mask]*1e9, pos[mask]*1e6 - pos[mask][0]*1e6, label='Displacement')
    plt.fill_between(time[mask]*1e9, pos[mask]*1e6 - pos[mask][0]*1e6 - (sigma_pos[mask]*1e6 - sigma_pos[mask][0]*1e6)/2, pos[mask]*1e6 - pos[mask][0]*1e6 + (sigma_pos[mask]*1e6 - sigma_pos[mask][0]*1e6)/2, color = 'r', alpha=0.5,label='Displacement Uncertainty')
    plt.xlabel("Time (ns)")
    plt.ylabel("Displacement (microns)")
    plt.legend()
    plt.savefig("displacement_w_uncert.png",dpi=300)
    plt.show()

    # mask2 = (time > 700e-9) & (time < 700.2e-9)
    # plt.scatter(real[mask2] * 1e3, imag[mask2] * 1e3, c=time[mask2]*1e9)
    # # plt.plot(real[mask2]*1e3, imag[mask2]*1e3,'k--', linewidth=0.5)
    # plt.xlabel("Real (mV)")
    # plt.ylabel("Imaginary (mV)")
    # plt.colorbar(label="Time (ns)")
    # # plt.savefig("Analytical.png", dpi=300)
    # plt.show()

    # a = 0.0001
    # b = 0.1
    # vf = 100
    # x = np.linspace(-750, 750, 1000)
    # z = a * x**2 + b * x
    # vc = vf / (2 * a * x + b)
    # plt.ylim([-5000,5000])
    # plt.scatter(x , vc)
    # plt.xlabel("X Position (microns)")
    # plt.ylabel("Closure Velocity (m/s)")
    # plt.axvline(-b/2/a,0,1,color='k',linestyle='--')
    # plt.savefig("closure_velocity_2d.png", dpi=300)
    # plt.show()

    A = np.array([[1,0.1],[0.1,0.5]])*1e-5
    B = np.array([1,0.5])*5e-3
    xv, yv = np.meshgrid(np.linspace(-750, 750, 100), np.linspace(-750, 750, 100))
    x = np.vstack((np.reshape(xv, xv.size), np.reshape(yv, xv.size))).T
    z = np.sum((x @ A) * x, axis=1) + x @ B
    zv = np.reshape(z, (int(np.sqrt(z.size)),int(np.sqrt(z.size))))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xv, yv, zv, cmap="coolwarm",
                        linewidth=0, antialiased=False)
    plt.xlabel("x (microns)")
    plt.ylabel("y (microns)")
    ax.set_zlabel("z (microns)")
    plt.savefig("3d_surface.png", dpi=300)
    plt.show()



