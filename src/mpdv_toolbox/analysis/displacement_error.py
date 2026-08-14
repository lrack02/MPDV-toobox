import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

def extract_analytical(voltage_df, time):
    """Extracts analytial signal for the time period where the alpss processor was conducted. Time may come from disp_df or other output"""
    t_start_idx = np.argwhere(time.values[0]==voltage_df["time"].values)[0][0]
    t_end_idx = np.argwhere(time.values[-1]==voltage_df["time"].values)[0][0]

    voltage = voltage_df.values[t_start_idx:t_end_idx+1,1:] #eliminate time column, end inclusive
    real = voltage[:, np.arange(0, voltage.shape[1]-1, step=2)] #real columns are even
    imag = voltage[:, np.arange(1, voltage.shape[1], step=2)] #imag columns are odd

    analytical = real + 1j * imag

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

def f_single_uncert(delta_theta, var_delta_theta):
    """
    Analytic (no sampling) variance of the step-function phase-unwrap correction
    f_i, computed from the marginal normal distribution of Delta_theta_i alone.

    f_i is +1 w.p. P(Delta_theta_i < -pi), -1 w.p. P(Delta_theta_i > pi), 0 otherwise.

    Returns
    -------
    array, shape (n_samples, n_channels)
        Variance of each individual f_i
    """
    F_minus = norm.cdf(-np.pi, loc=delta_theta, scale=np.sqrt(var_delta_theta))
    F_plus = norm.cdf(np.pi, loc=delta_theta, scale=np.sqrt(var_delta_theta))
    # f is +1 w.p. F_minus, -1 w.p. (1 - F_plus), 0 otherwise.
    return F_minus * (1 - F_minus) + F_plus * (1 - F_plus) + 2 * F_minus * (1 - F_plus)

def f_double_uncert(delta_theta, var_delta_theta, var_theta):
    """
    Analytic (no sampling) variance of f_i + f_{i-1}, the sum of adjacent
    phase-unwrap corrections, from the joint (bivariate) normal distribution of
    Delta_theta_i and Delta_theta_i-1. Looped explicitly over timesteps and
    probes so each pair's full derivation - marginal CDFs, the 9-region
    partition, then the variance of their sum - reads together in one place.

    Note: index 0 pairs the synthetic delta_theta[0] placeholder (see
    unwrap_phase/phas_uncert) with the real delta_theta[1]; phas_uncert
    overrides that one entry afterward rather than special-casing it here.

    Returns
    -------
    array, shape (n_samples-1, n_channels)
        Variance of each f_i + f_{i-1}
    """
    # Extract consecutive pairs
    dtheta_i = delta_theta[:-1]
    dtheta_im1 = delta_theta[1:]

    var_i = var_delta_theta[:-1]
    var_im1 = var_delta_theta[1:]
    cov_delta = -var_theta[:-1]

    vals = np.array([2, 1, 0, -1, -2])  # the values f_i + f_{i-1} can take

    n_pairs, n_channels = dtheta_i.shape
    var_f_double = np.zeros((n_pairs, n_channels))
    for t in range(n_pairs):
        for ch in range(n_channels):
            m0, v0 = dtheta_i[t, ch], var_i[t, ch]
            m1, v1 = dtheta_im1[t, ch], var_im1[t, ch]
            c = cov_delta[t, ch]

            cov_mat = np.array([[v0, c], [c, v1]])
            if not (np.isfinite(m0) and np.isfinite(m1) and np.all(np.isfinite(cov_mat))):
                continue  # leave var_f_double[t, ch] at 0
            if np.linalg.det(cov_mat) <= 0:
                continue

            # ========== MARGINAL CDFs ==========
            F_i_minus = norm.cdf(-np.pi, loc=m0, scale=np.sqrt(v0))
            F_i_plus = norm.cdf(np.pi, loc=m0, scale=np.sqrt(v0))
            F_im1_minus = norm.cdf(-np.pi, loc=m1, scale=np.sqrt(v1))
            F_im1_plus = norm.cdf(np.pi, loc=m1, scale=np.sqrt(v1))

            # ========== BIVARIATE CDF AT THE FOUR CORNERS ==========
            rv = multivariate_normal(mean=[m0, m1], cov=cov_mat)
            F_mm = rv.cdf([-np.pi, -np.pi])
            F_mp = rv.cdf([-np.pi, np.pi])
            F_pm = rv.cdf([np.pi, -np.pi])
            F_pp = rv.cdf([np.pi, np.pi])

            # ========== 9 REGIONS, one per (f_i, f_i-1) outcome ==========
            R1 = F_mm                                      # (+1,+1)
            R2 = F_pm - F_mm                                # (0,+1)
            R3 = F_im1_minus - F_pm                         # (-1,+1)
            R4 = F_mp - F_mm                                # (+1,0)
            R5 = F_pp - F_pm - F_mp + F_mm                  # (0,0)
            R6 = F_im1_plus - F_im1_minus - F_pp + F_pm     # (-1,0)
            R7 = F_i_minus - F_mp                           # (+1,-1)
            R8 = F_i_plus - F_i_minus - F_pp + F_mp         # (0,-1)
            R9 = 1 - F_i_plus - F_im1_plus + F_pp           # (-1,-1)

            probs = np.array([
                R1,            # sum = +2
                R2 + R4,       # sum = +1
                R3 + R5 + R7,  # sum = 0
                R6 + R8,       # sum = -1
                R9,            # sum = -2
            ])

            # ========== VARIANCE OF THE SUM ==========
            E_sum = np.sum(probs * vals)
            E_sum2 = np.sum(probs * vals**2)
            var_f_double[t, ch] = E_sum2 - E_sum**2

    return var_f_double

def phas_uncert(delta_theta, noise_frac_df):
    var_theta = noise_frac_df[noise_frac_df.columns[1:]].to_numpy()

    ### Compute variance of delta theta
    a = var_theta[:-1,:]
    b = var_theta[1:,:]
    var_delta_theta = a + b
    var_delta_theta = np.insert(var_delta_theta, 0 , 1e-10, axis=0)

    ### Step-function statistics needed for the cumulative phase variance below:
    ###   f_double[k] = Var(f_k + f_k+1)   (adjacent-pair variance)
    ###   f_single[k] = Var(f_k)           (single-index variance)
    var_f_single = f_single_uncert(delta_theta, var_delta_theta)
    var_f_double = f_double_uncert(delta_theta, var_delta_theta, var_theta)

    ### var double is equal to var single at the first timestep
    var_f_double[0] = var_f_single[1]

    ### Cumulative variance of S_i = f_0 + f_1 + ... + f_i, accounting for the one
    ### nonzero neighbor covariance Cov(f_i, f_i-1) at each step (f_i and f_j are
    ### independent for |i-j| >= 2, from the tri-diagonal delta_theta covariance model):
    ###   Var(S_i) = Var(S_i-1) + Var(f_i) + 2*Cov(f_i, f_i-1)
    ###            = Var(S_i-1) + [Var(f_i + f_i-1) - Var(f_i-1)]
    ###            = Var(S_i-1) + f_double[i-1] - f_single[i-1]
    increments = var_f_double - var_f_single[:-1]
    cum_var_f = var_f_single[0] + np.concatenate(
        [np.zeros((1, var_f_single.shape[1])), np.nancumsum(increments, axis=0)],
        axis=0
    )

    var_phas = var_theta + 4 * np.pi**2 * cum_var_f

    return var_theta, var_delta_theta, var_phas

def phas_uncert_independent(delta_theta, noise_frac_df):
    """
    Same as phas_uncert, but assumes samples are independent - i.e. it drops the
    Cov(f_i, f_i-1) correction and just sums the individual Var(f_i)'s:
    Var(S_i) = sum_{k<=i} Var(f_k), instead of accounting for the correlation
    between neighboring unwrap corrections. Provided for comparison against the
    exact (correlated) result from phas_uncert.
    """
    var_theta = noise_frac_df[noise_frac_df.columns[1:]].to_numpy()

    ### Compute variance of delta theta
    a = var_theta[:-1,:]
    b = var_theta[1:,:]
    var_delta_theta = a + b
    var_delta_theta = np.insert(var_delta_theta, 0 , 1e-10, axis=0)

    var_f_single = f_single_uncert(delta_theta, var_delta_theta)
    cum_var_f = np.nancumsum(var_f_single, axis=0)

    var_phas = var_theta + 4 * np.pi**2 * cum_var_f

    return var_theta, var_delta_theta, var_phas

def displacement_uncert(voltage_df, noise_frac_df, lam):
    time = noise_frac_df["time"]
    analytical = extract_analytical(voltage_df, time)

    theta, delta_theta, phas = unwrap_phase(analytical)
    var_theta, var_delta_theta, var_phas = phas_uncert(delta_theta, noise_frac_df)

    sigma_pos = lam / 2 / np.pi * np.sqrt(var_phas)

    sigma_disp_df = pd.DataFrame(data=np.column_stack((time, sigma_pos)), columns=noise_frac_df.columns)

    return sigma_disp_df


if __name__ == "__main__":
    from mpdv_toolbox.io.alpss import load_shot, filter_shot
    from matplotlib.colors import LinearSegmentedColormap

    base = r"C:\Users\lucas\OneDrive - Johns Hopkins\Ramesh Lab - Research\Papers\MPDV\MPDV_velocity_experiments\2026-06-30_velocity_vacuum\output_data\C1--JHAMAA00004_2026-06-30_18-27-46_shot13--00000"

    voltage_df = load_shot(base, "voltage")
    disp_df = load_shot(base, "displacement")
    vel_df = load_shot(base, "velocity--smooth")
    sigma_vel_df = load_shot(base, "veluncert")
    noise_frac_df = load_shot(base, "noisefrac")

    column_names = disp_df.columns

    time = disp_df["time"].to_numpy()

    analytical = extract_analytical(voltage_df, disp_df["time"])
    sigma_disp_df = displacement_uncert(voltage_df, noise_frac_df, 1550e-9)

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

    disp_df_filtered = filter_shot(disp_df, sigma_disp_df, 20e-6)
    sigma_disp_df_filtered = filter_shot(sigma_disp_df, sigma_disp_df, 20e-6)

    for probe in column_names[1:]:
        plt.plot(disp_df_filtered["time"]*1e9, disp_df_filtered[probe]*1e6, label=probe)
        plt.fill_between(sigma_disp_df_filtered["time"]*1e9, 
                         (disp_df_filtered[probe] - sigma_disp_df_filtered[probe])*1e6, 
                         (disp_df_filtered[probe] + sigma_disp_df_filtered[probe])*1e6,
                         alpha=0.5)
        plt.xlabel("time (ns)")
        plt.ylabel("displacement (microns)")
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


    # ### Plot bivariate distribution for delta_theta i and delta_theta i-1
    # idx = 40000
    # xx, yy = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 200), np.linspace(-2*np.pi, 2*np.pi, 200))
    # coords = np.dstack((xx, yy))  # shape (200, 200, 2): last axis is (delta_theta_i, delta_theta_i-1)

    # theta, delta_theta, phas = unwrap_phase(analytical)
    # var_theta = noise_frac_df[noise_frac_df.columns[1:]].to_numpy()
    # var_theta, var_delta_theta, var_phas = phas_uncert(delta_theta, noise_frac_df)

    # cov_matrix = np.array([
    #             [var_delta_theta[idx, 0], -var_theta[idx-1,0]],
    #             [-var_theta[idx-1,0], var_delta_theta[idx-1, 0]]
    #         ])
    # means = np.array([delta_theta[idx, 0], delta_theta[idx-1,0]])
    # rv = multivariate_normal(mean=means, cov=cov_matrix)

    # # n_samples_mc = int(1e6)
    # # samples = rv.rvs(n_samples_mc)

    # pdf_vals = rv.pdf(coords)  # shape (200, 200), matches xx/yy grid directly

    # # Sequential single-hue ramp (light -> dark), not a rainbow colormap: magnitude only.
    # blue_ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    # density_cmap = LinearSegmentedColormap.from_list("density_blue", blue_ramp)

    # fig, ax = plt.subplots()
    # mesh = ax.pcolormesh(xx, yy, pdf_vals, cmap=density_cmap, shading="auto")
    # fig.colorbar(mesh, ax=ax, label="Probability density")
    # ax.axvline(np.pi, color="#898781", linestyle="--", linewidth=1)
    # ax.axvline(-np.pi, color="#898781", linestyle="--", linewidth=1)
    # ax.axhline(np.pi, color="#898781", linestyle="--", linewidth=1)
    # ax.axhline(-np.pi, color="#898781", linestyle="--", linewidth=1)
    # ax.set_xlabel(r"$\Delta\theta_i$ (rad)")
    # ax.set_ylabel(r"$\Delta\theta_{i-1}$ (rad)")
    # ax.set_title(f"Bivariate PDF of successive phase increments")
    # ax.set_aspect("equal")
    # # plt.savefig("figures/Bivariate Normal Distribution.png", dpi=300)
    # plt.show()



    # var_f_sum, cov_f_sum, var_f_single, probs, vals = sampling_cov(delta_theta, var_delta_theta, var_theta)
    # plt.plot(var_f_sum[:,0])
    # plt.show()
    


    # # ### Plot single distribution for delta_theta i
    # # dt = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    # # pdf = norm.pdf(dt, loc=delta_theta[idx,0], scale=np.sqrt(noise_frac_df.iloc[idx, 1]))
    # # plt.plot(dt, pdf)
    # # plt.axvline(np.pi, color="#898781", linestyle="--", linewidth=1)
    # # plt.axvline(-np.pi, color="#898781", linestyle="--", linewidth=1)
    # # plt.xlim([-2 * np.pi, 2 * np.pi])
    # # plt.xlabel(r"$\Delta\theta_i$ (rad)")
    # # plt.ylabel(r"Probability Density")
    # # plt.savefig("figures/Normal Distribution.png", dpi=300)
    # # plt.show()
    




