"""Least-squares plane fitting across probe positions, used to quantify flyer tilt/planarity.

Plane model:  z = a*x + b*y + c
Normal:       n = [-a, -b, 1] / ||[-a, -b, 1]||
Tilt:         arccos(nz) in degrees
"""

from typing import NamedTuple

import numpy as np


class PlaneFit(NamedTuple):
    a: float
    b: float
    c: float
    nx: float
    ny: float
    nz: float
    tilt_deg: float
    r_squared: float
    rmse: float


def fit_plane(x, y, z):
    """Fit a plane to one time step of probe positions/displacements.

    x, y, z : array-like, one value per probe. NaNs are excluded. Returns a
    ``PlaneFit`` of all-NaN if fewer than 3 probes have finite values.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if valid.sum() < 3:
        return PlaneFit(*([np.nan] * 9))

    xv, yv, zv = x[valid], y[valid], z[valid]
    A = np.column_stack((xv, yv, np.ones(len(xv))))
    coeff, *_ = np.linalg.lstsq(A, zv, rcond=None)
    a, b, c = coeff

    zpred = A @ coeff
    ss_res = np.sum((zv - zpred) ** 2)
    ss_tot = np.sum((zv - zv.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(ss_res / len(zv))

    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    tilt_deg = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))

    return PlaneFit(a, b, c, normal[0], normal[1], normal[2], tilt_deg, r2, rmse)


def fit_plane_batch(x, y, z_matrix):
    """Vectorized ``fit_plane`` across many time steps at once.

    x, y : array-like, shape (Nprobes,), assumed finite for all probes.
    z_matrix : array-like, shape (Nt, Nprobes), assumed finite for every entry.

    Returns arrays (a, b, c, nx, ny, nz, tilt_deg, r_squared, rmse), each shape (Nt,).
    """
    x, y, z_matrix = np.asarray(x), np.asarray(y), np.asarray(z_matrix)
    A = np.column_stack((x, y, np.ones(len(x))))
    pinv_A = np.linalg.pinv(A)
    coeffs = (pinv_A @ z_matrix.T).T  # (Nt, 3)

    normals = np.column_stack([-coeffs[:, 0], -coeffs[:, 1], np.ones(len(coeffs))])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    zpred = (A @ coeffs.T).T
    zmean = z_matrix.mean(axis=1, keepdims=True)
    ss_res = np.sum((z_matrix - zpred) ** 2, axis=1)
    ss_tot = np.sum((z_matrix - zmean) ** 2, axis=1)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    rmse = np.sqrt(ss_res / z_matrix.shape[1])
    tilt_deg = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))

    return (coeffs[:, 0], coeffs[:, 1], coeffs[:, 2],
            normals[:, 0], normals[:, 1], normals[:, 2],
            tilt_deg, r2, rmse)
