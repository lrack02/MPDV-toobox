import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

from mpdv_toolbox.io.alpss import load_probe_positions


def surface_fit(base, disp_df, positions_csv, order=2, focus_scale=2.0, output_csv=None):
    """
    Linear regression fitting on the displacement data using the -displacement.csv files.

    Args: 
        base: base pdv filename including folder path without the -displacement.csv 
        disp_df: displacement dataframe, time corrected. 
        positions_csv: path to pdv spatial positions csv
        order: polynomial fit order (0 = constant, 1 = linear, 2 = quadratic, "2_symmetric" = quadratic with A constrained to a scalar multiple of the identity; all include a constant offset C)
        focus_scale: multiplier to decrease the probe spacing
        output_csv: boolean: True or False. If True, saves to output_data directory with -fit_popt.csv and -fit_pcov.csv
    """

    probe_locs = load_probe_positions(positions_csv, focus_scale=focus_scale)

    time_values = disp_df["time"].values

    # Define x data
    x_values = probe_locs['x_position'].values / 1e6 #convert from microns to meters
    y_values = probe_locs['y_position'].values / 1e6 #convert from microns to meters
    x_data = np.vstack((x_values, y_values)).T

    # Order the probes
    disp_df_ordered = pd.DataFrame([])
    for probe_num in probe_locs["probe_number"].values:
        probe_label = "probe_" + str(probe_num)
        disp_df_ordered[probe_label] = disp_df[probe_label]

    # Verify ordering
    print("Probe numbers:", probe_locs["probe_number"].values)
    print("Columns:", disp_df_ordered.columns.values)
    print("X data shape:", x_data.shape)

    # Initial guess
    if order == 0:
        p0 = (0,)
    elif order == 1:
        p0 = (0, 0, 0)
    elif order == "2_symmetric":
        p0 = (0, 0, 0, 0)
    elif order == 2:
        p0 = (0, 0, 0, 0, 0, 0)
    else:
        raise ValueError(f"Order {order} not implemented")

    popt_array = []
    pcov_array = []
    
    for idx, t in enumerate(time_values):
        y_values_at_t = disp_df_ordered.iloc[idx, :].values  # Use .iloc for clarity
        
        # Filter NaN
        mask = ~np.isnan(y_values_at_t) #TODO: include loading the uncertainty data
        x_data_masked = x_data[mask]
        y_data_masked = y_values_at_t[mask]
                
        # Skip if insufficient data
        if y_data_masked.size < len(p0) + 1:  # Need at least p0 + 1 points
            popt_array.append(np.full(len(p0), np.nan))
            pcov_array.append(np.full(len(p0) * len(p0), np.nan))
            continue

        try:
            popt, pcov = curve_fit(
                surface,
                xdata=x_data_masked,
                ydata=y_data_masked,
                p0=p0
            )
            p0 = popt
            popt_array.append(popt)
            pcov_array.append(pcov.flatten())
            # print(f"  → Fit successful")
        except RuntimeError as e:
            print(f"  → Fit failed: {e}")
            popt_array.append(np.full(len(p0), np.nan))
            pcov_array.append(np.full(len(p0) * len(p0), np.nan))
            continue

    # save to csv
    popt_df = pd.DataFrame(popt_array)
    pcov_df = pd.DataFrame(pcov_array)

    if order == 0:
        popt_df.columns = ["C"]
        pcov_df.columns = ["C-C"]

    elif order == 1:
        popt_df.columns = ["B1", "B2", "C"]
        pcov_df.columns = ["B1-B1", "B1-B2", "B1-C",
                           "B2-B1", "B2-B2", "B2-C",
                           "C-B1", "C-B2", "C-C"]
    elif order == "2_symmetric":
        popt_df.columns = ["A11", "B1", "B2", "C"]
        pcov_df.columns = ["A11-A11", "A11-B1", "A11-B2", "A11-C",
                           "B1-A11", "B1-B1", "B1-B2", "B1-C",
                           "B2-A11", "B2-B1", "B2-B2", "B2-C",
                           "C-A11", "C-B1", "C-B2", "C-C"]
    elif order == 2:
        popt_df.columns = ["A11", "A12", "A22", "B1", "B2", "C"]
        pcov_df.columns = ["A11-A11", "A11-A12", "A11-A22", "A11-B1", "A11-B2", "A11-C",
                           "A12-A11", "A12-A12", "A12-A22", "A12-B1", "A12-B2", "A12-C",
                           "A22-A11", "A22-A12", "A22-A22", "A22-B1", "A22-B2", "A22-C",
                           "B1-A11", "B1-A12", "B1-A22", "B1-B1", "B1-B2", "B1-C",
                           "B2-A11", "B2-A12", "B2-A22", "B2-B1", "B2-B2", "B2-C",
                           "C-A11", "C-A12", "C-A22", "C-B1", "C-B2", "C-C"]

    popt_df.to_csv(base+"-fit_popt.csv")
    pcov_df.to_csv(base+"-fit_pcov.csv")
    
    return popt_df, pcov_df

def surface(x, *p):
    """
    z = C (constant)
    OR
    z = x^T B + C (linear)
    OR
    z = x^T A x + x^T B + C, A = a*I (symmetric quadratic)
    OR
    z = x^T A x + x^T B + C (quadratic)

    Args:
        x: shape (n_samples, 2)
        order: order of taylor polynomial
        p: fitting parameters based on the order of function. p = len(1) if order=0, len(3) if order=1, len(4) if order="2_symmetric", len(6) if order=2.
    """
    if len(p) == 1:
        # Constant case, order = 0. p = C
        # print("constant fit")
        z = np.full(x.shape[0], p[0])
    elif len(p) == 3:
        # Linear case, order = 1. p = B1, B2, C
        # print("linear fit")
        B = np.array([p[0],p[1]])
        C = p[2]
        z = x @ B + C
    elif len(p) == 4:
        # Symmetric quadratic case. A is assumed to be some constant times identity
        A = np.array([[p[0], 0],[0, p[0]]])
        B = np.array([p[1],p[2]])
        C = p[3]
        z = np.sum((x @ A) * x, axis=1) + x @ B + C
    elif len(p) == 6:
        # Quadratic case, order = 2. p = A00, A01, A11, B1, B2, C
        # print("quadratic fit")
        A = np.array([[p[0], p[1]],[p[1], p[2]]])
        B = np.array([p[3],p[4]])
        C = p[5]
        z = np.sum((x @ A) * x, axis=1) + x @ B + C
    else:
        print("Polynomial Order Not Implimented")
        z = None
    return z

# surface_fit(base=r"C:\Users\lucas\OneDrive - Johns Hopkins\Ramesh Lab - Research\Papers\MPDV\MPDV_velocity_experiments\2026-06-30_velocity_vacuum\output_data\C1--JHAMAA00004_2026-06-30_18-24-54_shot01--00000",
#             positions_csv=r"C:\Users\lucas\OneDrive - Johns Hopkins\Ramesh Lab - Research\Papers\MPDV\Spatial_Distribution\2026-02-18\positions.csv")