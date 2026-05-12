import json
import os
import pickle

import h5py
import numpy as np
import pyfftw
from numba import njit, prange


@njit(parallel=True)
def _accumulate_bins(k_mag_flat, cov_k_flat, shot_flat, k_bins, n_bins, do_shot):
    """
    Accumulate the covariance and shot noise values into the specified wavenumber bins.

    Parameters:
    ----------
    k_mag_flat: 1D array
        Flattened array of the magnitudes of the wavevectors corresponding to each Fourier mode.
    cov_k_flat: 1D array
        Flattened array of the covariance values (real part of the product of the Fourier transforms) corresponding to each Fourier mode.
    shot_flat: 1D array
        Flattened array of the shot noise correction values corresponding to each Fourier mode, calculated from the shot noise kernel if provided.
    k_bins: 1D array
        Array of the edges of the wavenumber bins to accumulate the values into.
    n_bins: int
        Number of wavenumber bins, which should be equal to len(k_bins) - 1.
    do_shot: bool
        Whether to accumulate the shot noise correction values. This should be True if a shot noise kernel was provided and the power spectrum being calculated is an auto-power spectrum (i.e., delta_k_2 is None), and False otherwise.

    Returns:
    -------
    counts: 1D array
        Array of the counts of Fourier modes that fall into each wavenumber bin.
    sum_cov: 1D array
        Array of the sums of the covariance values for the Fourier modes that fall into each wavenumber bin.
    sum_shot: 1D array
        Array of the sums of the shot noise correction values for the Fourier modes that fall into each wavenumber bin. This will be an array of zeros if do_shot is False.
    """

    counts = np.zeros(n_bins)
    sum_cov = np.zeros(n_bins)
    sum_shot = np.zeros(n_bins)

    for i in prange(len(k_mag_flat)):
        k = k_mag_flat[i]

        lo, hi = 0, n_bins
        while lo < hi:
            mid = (lo + hi) // 2
            if k_bins[mid] < k:
                lo = mid + 1
            else:
                hi = mid
        b = lo - 1

        if 0 <= b < n_bins:
            counts[b] += 1
            sum_cov[b] += cov_k_flat[i]
            if do_shot:
                sum_shot[b] += shot_flat[i]

    return counts, sum_cov, sum_shot


class FFTWPlan:
    def __init__(self, N_cell, n_threads=None, fftw_effort="FFTW_MEASURE"):
        """
        Initialize the FFTW plan with the number of cells, number of threads, and effort level for optimization.

        Parameters:
        ----------
        N_cell: int
            Number of cells in each dimension for the FFT grid.
        n_threads: int, optional
            Number of threads to use for the FFTW library. If None, it will use the number of CPU cores available. The default value is None.
        fftw_effort: str, optional
            Effort level for the FFTW library, which determines the amount of time spent optimizing the FFT plan. The default value is "FFTW_MEASURE". The available options are "FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", and "FFTW_EXHAUSTIVE", with increasing levels of optimization and time spent on planning.
        """

        self.N_cell = N_cell
        self.n_threads = n_threads or os.cpu_count()
        self.fftw_effort = fftw_effort

    def build_forward_plan(self):
        """
        Build the FFTW plan for the forward Fourier transform. The input array will be a real-valued 3D array of shape (N_cell, N_cell, N_cell), and the output array will be a complex-valued 3D array of shape (N_cell, N_cell, N_cell//2 + 1) due to the use of the rfftn function for real-to-complex transforms.

        Returns:
        -------
        fftw_forward: pyfftw.FFTW object
            FFTW plan for the forward Fourier transform.
        """

        _in = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell), dtype="float64"
        )
        _out = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell // 2 + 1), dtype="complex128"
        )

        return pyfftw.FFTW(
            _in,
            _out,
            axes=(0, 1, 2),
            direction="FFTW_FORWARD",
            flags=(self.fftw_effort,),
            threads=self.n_threads,
        )

    def build_inverse_plan(self):
        """
        Build the FFTW plan for the inverse Fourier transform. The input array will be a complex-valued 3D array of shape (N_cell, N_cell, N_cell//2 + 1), and the output array will be a real-valued 3D array of shape (N_cell, N_cell, N_cell). The inverse transform will be normalized by 1/N to match the normalization convention used by numpy's FFT functions.

        Returns:
        -------
        fftw_inverse: pyfftw.FFTW object
            FFTW plan for the inverse Fourier transform, with normalization to match numpy's convention.
        """

        _in_inv = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell // 2 + 1), dtype="complex128"
        )
        _out_inv = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell), dtype="float64"
        )

        return pyfftw.FFTW(
            _in_inv,
            _out_inv,
            axes=(0, 1, 2),
            direction="FFTW_BACKWARD",
            flags=(self.fftw_effort,),
            threads=self.n_threads,
            normalise_idft=True,
        )

    def save_wisdom(self, filename):
        """
        Save the FFTW wisdom containing the optimized plans for the FFTW library, which can be reused in future runs to speed up the planning phase of the Fourier transforms.

        Parameters:
        ----------
        filename: str
            Name of the pickle file to save the FFTW wisdom to.
        """

        wisdom = pyfftw.export_wisdom()
        with open(filename, "wb") as f:  # noqa: PTH123
            pickle.dump(wisdom, f)

    def load_wisdom(self, filename):
        """
        Load the FFTW wisdom from a pickle file and import it into the FFTW library to reuse the optimized plans for the Fourier transforms.

        Parameters:
        ----------
        filename: str
            Name of the pickle file to load the FFTW wisdom from.
        """

        with open(filename, "rb") as f:  # noqa: PTH123
            wisdom = pickle.load(f)
        pyfftw.import_wisdom(wisdom)


class FourierTransform:
    """
    Class for performing Fourier transform and calculating power spectrum from a 3D field.
    """

    def __init__(self, N_cell, L_box, N_part, fftw_forward, fftw_inverse=None):
        """
        Initialize the Fourier transform with the field, number of cells, and box size.

        Parameters:
        ----------
        N_cell: int
            Number of cells in each dimension.
        L_box: float
            Size of the box in real space.
        N_part: int
            Total number of particles in the simulation.
        fftw_forward: pyfftw.FFTW
            FFTW plan for the forward Fourier transform.
        fftw_inverse: pyfftw.FFTW, optional
            FFTW plan for the inverse Fourier transform. The default value is None, with which no inverse transform will be performed.
        """

        self._N_cell = N_cell
        self._L_box = L_box
        self._N_part = N_part
        self._dx, self._k_min, self._k_max, self._P_shot = self._get_basic_params(
            N_cell, L_box, N_part
        )

        self._fftw_forward = fftw_forward
        self._fftw_inverse = fftw_inverse

    def _get_basic_params(self, N_cell, L_box, N_part):
        """
        Get the basic parameters for the Fourier transform, including the grid spacing, minimum wavenumber, and maximum wavenumber.

        Parameters:
        ----------
        N_cell: int
            Number of cells in each dimension.
        L_box: float
            Size of the box in real space.
        N_part: int
            Total number of particles in the simulation.

        Returns:
        -------
        dx: float
            Grid spacing in real space.
        k_min: float
            Minimum wavenumber corresponding to the fundamental mode of the box.
        k_max: float
            Maximum wavenumber corresponding to the Nyquist frequency of the grid.
        P_shot: float
            Shot noise power spectrum, calculated as the volume of the box divided by the total number of particles.
        """

        dx = L_box / N_cell
        k_min = 2 * np.pi / L_box
        k_max = np.pi / dx

        P_shot = L_box**3 / N_part

        return dx, k_min, k_max, P_shot

    def get_fourier_transform(self, delta_x):
        """
        Get the Fourier transform of the input field and calculate the magnitude of the wavevector for each Fourier mode.

        Parameters:
        ----------
        delta_x: 3D array
            3D field in real space.

        Returns:
        -------
        k_mag: 3D array
            Magnitude of the wavevector corresponding to each Fourier mode.
        delta_k: 3D array
            Fourier transform of the input field.
        """

        self._fftw_forward.input_array[:] = delta_x
        self._fftw_forward()

        delta_k = self._fftw_forward.output_array.copy()

        _k_1d = np.fft.fftfreq(self._N_cell, d=self._dx) * 2 * np.pi
        _kz_1d = np.fft.rfftfreq(self._N_cell, d=self._dx) * 2 * np.pi

        _kx = _k_1d[:, None, None]
        _ky = _k_1d[None, :, None]
        _kz = _kz_1d[None, None, :]

        k_mag = np.sqrt(_kx**2 + _ky**2 + _kz**2)

        return k_mag, delta_k

    def save_fourier_transform(self, k_mag, delta_k, filename, save_k_mag=True):
        """
        Save the Fourier transform and the corresponding wavevector magnitudes to an HDF5 file.

        Parameters:
        ----------
        k_mag: 3D array
            Magnitude of the wavevector corresponding to each Fourier mode.
        delta_k: 3D array
            Fourier transform of the input field.
        filename: str
            Name of the HDF5 file to save the data to.
        save_k_mag: bool, optional
            Whether to save the wavevector magnitudes in the HDF5 file.
        """

        with h5py.File(filename, "w") as f:
            if save_k_mag:
                f.create_dataset("WaveVector", data=k_mag)
            f.create_dataset("FourierTransform", data=delta_k)

    def inv_fourier_transform(self, delta_k):
        """
        Perform the inverse Fourier transform to get back the field in real space from its Fourier transform.

        Parameters:
        ----------
        delta_k: 3D array
            Fourier transform of the field.

        Returns:
        -------
        delta_x: 3D array
            Field in real space obtained from the inverse Fourier transform of the input Fourier transform.
        """

        if self._fftw_inverse is None:
            raise ValueError(  # noqa: TRY003
                "Inverse FFTW plan is not available. Please build the inverse plan first."  # noqa: EM101
            )

        self._fftw_inverse.input_array[:] = delta_k
        self._fftw_inverse()

        return self._fftw_inverse.output_array.copy()

    def get_conv_kernels(self, p=2):
        """
        Get the convolution kernels for the Fourier transform, which includes both the shot noise correction and the deconvolution kernel for the mass assignment scheme.

        Parameters:
        ----------
        p: int, optional
            Order of the mass assignment scheme. The deconvolution kernel will be calculated as the sinc function raised to the power of p for each dimension. The default value is 2, corresponding to the Cloud-in-Cell (CIC) mass assignment scheme.

        Returns:
        -------
        shot_noise_kernel: 3D array
            Shot noise correction kernel in Fourier space following Jing (2005), calculated as the product of the single-shot noise kernels for each dimension.
        deconv_kernel: 3D array
            Deconvolution kernel in Fourier space, calculated as the product of the single-dimension deconvolution kernels for each dimension.
        """

        _k_1d = np.fft.fftfreq(self._N_cell, d=self._dx) * 2 * np.pi
        _kz_1d = np.fft.rfftfreq(self._N_cell, d=self._dx) * 2 * np.pi

        _kx = _k_1d[:, None, None]
        _ky = _k_1d[None, :, None]
        _kz = _kz_1d[None, None, :]

        def _single_shot_noise_kernel(k_1d):
            return 1 - 2 / 3 * np.sin(k_1d * self._dx / 2) ** 2

        shot_noise_kernel = (
            _single_shot_noise_kernel(_kx)
            * _single_shot_noise_kernel(_ky)
            * _single_shot_noise_kernel(_kz)
        )

        def _single_deconv_kernel(k_1d):
            return np.sinc(k_1d * self._dx / (2 * np.pi)) ** p

        deconv_kernel = (
            _single_deconv_kernel(_kx)
            * _single_deconv_kernel(_ky)
            * _single_deconv_kernel(_kz)
        )

        return shot_noise_kernel, deconv_kernel

    def save_conv_kernels(self, shot_noise_kernel, deconv_kernel, filename):
        """
        Save the convolution kernels to an HDF5 file.

        Parameters:
        ----------
        shot_noise_kernel: 3D array
            Shot noise correction kernel in Fourier space.
        deconv_kernel: 3D array
            Deconvolution kernel in Fourier space.
        filename: str
            Name of the HDF5 file to save the kernels to.
        """

        with h5py.File(filename, "w") as f:
            f.create_dataset("ShotNoiseKernel", data=shot_noise_kernel)
            f.create_dataset("DeconvKernel", data=deconv_kernel)

    def get_power_spectrum(
        self,
        k_mag,
        delta_k_1,
        delta_k_2=None,
        shot_noise_kernel=None,
        n_k_bins=31,
        P_k_11_raw=None,
        P_k_22_raw=None,
    ):
        """
        Calculate the power spectrum from the Fourier transform of the field. If a second Fourier transform is provided, calculate the cross-power spectrum between the two fields.

        Parameters:
        ----------
        k_mag: 3D array
            Magnitude of the wavevector corresponding to each Fourier mode, obtained from the Fourier transform of the field.
        delta_k_1: 3D array
            Fourier transform of the first field.
        delta_k_2: 3D array, optional
            Fourier transform of a second field. If provided, the cross-power spectrum between the two fields will be calculated. If None, the auto-power spectrum of the first field will be calculated.
        shot_noise_kernel: 3D array, optional
            Shot noise correction kernel in Fourier space. If provided, it will be used to correct the power spectrum for shot noise. If not provided, no shot noise correction will be applied.
        n_k_bins: int, optional
            Number of bins to use for the power spectrum. The wavenumber range will be divided into this many logarithmically spaced bins.
        P_k_11_raw: 1D array, optional
            Raw power spectrum values for the first field, used for calculating the uncertainty in the power spectrum. If not provided, it will be taken as the raw auto-power spectrum of the first field.
        P_k_22_raw: 1D array, optional
            Raw power spectrum values for the second field, used for calculating the uncertainty in the power spectrum. If not provided, it will be taken as the raw auto-power spectrum of the first field.

        Returns:
        -------
        k_bin_centers: 1D array
            Centers of the wavenumber bins used for the power spectrum.
        P_k: 1D array
            Power spectrum values corresponding to the wavenumber bins.
        P_k_err: 1D array
            Uncertainties in the power spectrum values, calculated as the standard error of the mean for each bin.
        """

        _delta_k_1 = delta_k_1
        _delta_k_2 = delta_k_2 if delta_k_2 is not None else delta_k_1

        _k_mag = np.ascontiguousarray(k_mag.ravel())
        _cov_k = np.ascontiguousarray(np.real(_delta_k_1 * np.conj(_delta_k_2)).ravel())

        _do_shot = shot_noise_kernel is not None and delta_k_2 is None
        _shot_k = (
            np.ascontiguousarray(shot_noise_kernel.ravel())
            if shot_noise_kernel
            else np.empty(0)
        )

        _k_bins = np.logspace(np.log10(self._k_min), np.log10(self._k_max), n_k_bins)
        k_bin_centers = 0.5 * (_k_bins[:-1] + _k_bins[1:])
        _n_bins = len(k_bin_centers)

        _counts, _sum_cov_k, _sum_shot_k = _accumulate_bins(
            _k_mag, _cov_k, _shot_k, _k_bins, _n_bins, _do_shot
        )

        _mask = _counts > 0
        P_k_raw = np.zeros_like(k_bin_centers)
        P_k_raw[_mask] = _sum_cov_k[_mask] / _counts[_mask]
        P_k_raw *= (self._L_box**3) / (self._N_cell**6)

        _P_shot_k = np.zeros_like(k_bin_centers)
        _P_shot_k[_mask] = _sum_shot_k[_mask] / _counts[_mask]
        _P_shot_k *= self._P_shot
        P_k = P_k_raw - _P_shot_k

        P_k_err = np.zeros_like(k_bin_centers)
        _P_k_11_raw = P_k_11_raw if P_k_11_raw is not None else P_k_raw
        _P_k_22_raw = P_k_22_raw if P_k_22_raw is not None else P_k_raw
        P_k_err[_mask] = np.sqrt(
            1
            / _counts[_mask]
            * (P_k_raw[_mask] ** 2 + _P_k_11_raw[_mask] * _P_k_22_raw[_mask])
        )

        return k_bin_centers, P_k_raw, P_k, P_k_err

    def save_power_spectrum(self, k_bins, P_k_raw, P_k, P_k_err, filename):
        """
        Save the power spectrum and its uncertainties to an json file.

        Parameters:
        ----------
        k_bins: 1D array
            Centers of the wavenumber bins used for the power spectrum.
        P_k_raw: 1D array
            Raw power spectrum values corresponding to the wavenumber bins, before shot noise subtraction.
        P_k: 1D array
            Power spectrum values corresponding to the wavenumber bins.
        P_k_err: 1D array
            Uncertainties in the power spectrum values.
        filename: str
            Name of the JSON file to save the data to.
        """

        with open(filename, "w") as f:  # noqa: PTH123
            json.dump(
                {
                    "k_bins": k_bins.tolist(),
                    "P_k_raw": P_k_raw.tolist(),
                    "P_k": P_k.tolist(),
                    "P_k_err": P_k_err.tolist(),
                },
                f,
            )
