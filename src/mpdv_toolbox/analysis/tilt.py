import numpy as np


def min_max_closure_velocity(popt_df, v_f = 800, t_eval=200e-9, r=750e-6):
    """
    Returns the minimum and maximum closure front velocity at the flyer edge for a given flyer radius.

    Assumes symmetric quadratic taylor expansion ("2_symmetric" fit) where A is a 
    diagonal matrix where A11=A22 and A12=0.

    popt_df: dataframe from the fitting solution
    v_f: flyer impact velocity (m/s)
    t_eval: flyer impact time (s)
    r: flyer radius (m)
    
    """
    t_eval_idx = np.argmin(np.abs(popt_df["time"] - t_eval))

    A11 = popt_df["A11"][t_eval_idx]
    B1 = popt_df["B1"][t_eval_idx]
    B2 = popt_df["B2"][t_eval_idx]

    thetas = np.array([np.atan(B2/B1), np.atan(B2/B1)+np.pi])

    v_c = v_f / np.sqrt((A11 * r * np.cos(thetas) + B1)**2+(A11 * r * np.sin(thetas) + B2)**2)

    min_v_c = np.min(v_c)
    max_v_c = np.max(v_c)

    return min_v_c, max_v_c

def tilt_vs_time(popt_df):
    """ 
    returns the stereographic projection of the flyer (phi: azimuthal angle, theta: x-y plane projection angle) 
    tilt as an array of same length as time

    popt_df: dataframe from the fitting solution
    """
    grad = popt_df[["B1", "B2"]].values
    phi = np.arctan(np.linalg.norm(grad, axis = 1))
    theta = np.arctan2(grad[:,1], grad[:,0])
    return phi, theta

def tilt_at_time(popt_df, t_eval):
    """ 
    returns the stereographic projection of the flyer (phi: azimuthal angle, theta: x-y plane projection angle) 
    tilt at a specific time t_eval

    popt_df: dataframe from the fitting solution
    t_eval: evaluation time
    """
    t_eval_idx = np.argmin(np.abs(popt_df["time"] - t_eval))

    grad = popt_df[["B1", "B2"]].values[t_eval_idx]
    phi = np.arctan(np.linalg.norm(grad))
    theta = np.arctan2(grad[1], grad[0])

    return phi, theta