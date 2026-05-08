import json

import h5py
import numpy as np


class FourierTransform:
    """
    Class for performing Fourier transform and calculating power spectrum from a 3D field.
    """

    def __init__(self, N_cell, L_box, N_part):
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
        """

        self._N_cell = N_cell
        self._L_box = L_box
        self._N_part = N_part
        self._dx, self._k_min, self._k_max, self._P_shot = self._get_basic_params(
            N_cell, L_box, N_part
        )

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

    def fourier_transform(self, delta_x):
        """
        Perform the Fourier transform of the input field and calculate the magnitude of the wavevector for each Fourier mode.

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

        delta_k = np.fft.fftn(delta_x)

        _k_1d = np.fft.fftfreq(self._N_cell, d=self._dx) * 2 * np.pi

        _kx = _k_1d[:, None, None]
        _ky = _k_1d[None, :, None]
        _kz = _k_1d[None, None, :]

        k_mag = np.sqrt(_kx**2 + _ky**2 + _kz**2)

        return k_mag, delta_k

    def save_fourier_transform(self, k_mag, delta_k, filename):
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
        """

        with h5py.File(filename, "w") as f:
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

        return np.fft.ifftn(delta_k).real

    def get_power_spectrum(
        self,
        k_mag,
        delta_k_1,
        delta_k_2=None,
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

        _k_mag = k_mag.flatten()
        _cov_k = np.real(_delta_k_1 * np.conj(_delta_k_2)).flatten()

        _k_bins = np.logspace(np.log10(self._k_min), np.log10(self._k_max), n_k_bins)
        k_bin_centers = 0.5 * (_k_bins[:-1] + _k_bins[1:])

        _counts, _ = np.histogram(_k_mag, bins=_k_bins)
        _sum_cov_k, _ = np.histogram(_k_mag, bins=_k_bins, weights=_cov_k)

        _mask = _counts > 0
        _P_k_raw = np.zeros_like(k_bin_centers)
        _P_k_raw[_mask] = _sum_cov_k[_mask] / _counts[_mask]
        _P_k_raw *= (self._L_box**3) / (self._N_cell**6)

        P_k = _P_k_raw - self._P_shot if delta_k_2 is None else _P_k_raw

        P_k_err = np.zeros_like(k_bin_centers)
        _P_k_11_raw = P_k_11_raw if P_k_11_raw is not None else _P_k_raw
        _P_k_22_raw = P_k_22_raw if P_k_22_raw is not None else _P_k_raw
        P_k_err[_mask] = np.sqrt(
            1
            / _counts[_mask]
            * (_P_k_raw[_mask] ** 2 + _P_k_11_raw[_mask] * _P_k_22_raw[_mask])
        )

        return k_bin_centers, P_k, P_k_err

    def save_power_spectrum(self, k_bins, P_k, P_k_err, filename):
        """
        Save the power spectrum and its uncertainties to an json file.

        Parameters:
        ----------
        k_bins: 1D array
            Centers of the wavenumber bins used for the power spectrum.
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
                    "P_k": P_k.tolist(),
                    "P_k_err": P_k_err.tolist(),
                },
                f,
            )
