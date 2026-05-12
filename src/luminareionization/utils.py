import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline


def get_interp(x_bins, y_bins):
    """
    Get the interpolated profiles of the halo.

    Parameters:
    ----------
    x_bins: array
        Array of the x-param.
    y_bins: array
        Array of the y-param.

    Returns:
    -------
    interp: CubicSpline
        Interpolated profile of the halo.
    """

    _x_order = np.argsort(x_bins)
    _x_increasing_mask = np.append([True], np.diff(x_bins[_x_order]) > 0)

    x_bins = x_bins[_x_order][_x_increasing_mask]
    y_bins = y_bins[_x_order][_x_increasing_mask]

    _finite_mask = np.logical_and(np.isfinite(x_bins), np.isfinite(y_bins))

    return CubicSpline(x_bins[_finite_mask], y_bins[_finite_mask])


@njit(parallel=True)
def get_bias_factor(k_mag, b0, k0, alpha, beta=2):
    """
    Compute the bias factor for a given k magnitude.

    Parameters:
    ----------
    k_mag: array
        Array of k magnitudes.
    b0: float
        Bias factor at k=0.
    k0: float
        Scale parameter for the bias factor.
    alpha: float
        Power-law index for the bias factor.
    beta: float, optional
        Smoothing parameter for the bias factor. Default is 2.

    Returns:
    -------
    b_k: array
        Array of bias factors corresponding to the input k magnitudes.
    """

    n = k_mag.size
    b_k = np.empty(n)

    for i in prange(n):
        b_k[i] = b0 / (1 + (k_mag[i] / k0) ** beta) ** (alpha / beta)

    return b_k


@njit(parallel=True)
def get_rms_err(y, y_hat):
    """
    Compute the root mean square error between the true and predicted values.

    Parameters:
    ----------
    y: array
        Array of the true values.
    y_hat: array
        Array of the predicted values.

    Returns:
    -------
    rms_err: float
        Root mean square error between the true and predicted values.
    """

    n = y.size
    err_sum = 0.0

    for i in prange(n):
        err_sum += (1 - y_hat[i] / y[i]) ** 2

    return np.sqrt(err_sum / n)
