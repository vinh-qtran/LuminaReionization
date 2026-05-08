import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2


class BaseFitter:
    """
    Base class for fitting a model to data using least squares optimization.
    """

    def __init__(self, x, y, w=None):
        """
        Initialize the fitter with data and optional weights.

        Parameters:
        ----------
        x: array-like
            Independent variable data.
        y: array-like
            Dependent variable data.
        w: array-like, optional
            Weights for the data points. If None, all points are equally weighted.
        """

        self._x = x
        self._y = y
        self._w = w

    def _get_model(self, x, params):
        """
        Get the model predictions for given parameters.

        Parameters:
        ----------
        x: array-like
            Independent variable data.
        params: array-like
            Model parameters.

        Returns:
        -------
        yhat: array-like
            Model predictions for the given parameters.
        """

        raise NotImplementedError("Not implemented in base class.")  # noqa: EM101

    def _get_residuals(self, params):
        """
        Calculate the residuals between the observed data and the model predictions for given parameters.

        Parameters:
        ----------
        params: array-like
            Model parameters.

        Returns:
        -------
        residuals: array-like
            Residuals between the observed data and the model predictions, weighted by the square root of the weights if provided.
        """

        _yhat = self._get_model(self._x, params)
        _w = self._w if self._w is not None else np.ones_like(self._y)

        return (self._y - _yhat) * np.sqrt(_w)

    def _get_chisqr(self, params):
        """
        Get the chi-squared value for the given parameters.

        Parameters:
        ----------
        params: array-like
            Model parameters.

        Returns:
        -------
        chisqr: float
            Chi-squared value for the given parameters.
        """

        return np.sum(self._get_residuals(params) ** 2)

    def _get_initial_guess(self):
        """
        Get the initial guess for the model parameters.

        Returns:
        -------
        initial_guess: array-like
            Initial guess for the model parameters.
        """

        raise NotImplementedError("Not implemented in base class.")  # noqa: EM101

    def fit(self, bounds=(-np.inf, np.inf)):
        """
        Perform the least squares fitting to find the best-fit parameters.

        Parameters:
        ----------
        bounds: tuple of array-like, optional
            Lower and upper bounds on the parameters. Each array-like should have the same shape as the number of parameters. If not provided, the parameters are unbounded.

        Returns:
        -------
        fit_result: dict
            A dictionary containing the best-fit parameters, their uncertainties, chi-squared values, and other relevant information about the fit.
        """

        _initial_guess = self._get_initial_guess()
        _result = least_squares(self._get_residuals, _initial_guess, bounds=bounds)

        _chisqr = self._get_chisqr(_result.x)
        _alpha = 1 - chi2.cdf(_chisqr, len(self._x) - len(_result.x))

        try:
            _cov = np.linalg.inv(np.dot(_result.jac.T, _result.jac))
            _params_err = np.sqrt(np.diagonal(_cov))
        except np.linalg.LinAlgError:
            _cov = None
            _params_err = None

        return {
            "params": _result.x,
            "params_err": _params_err,
            "chisqr": _chisqr,
            "reduced_chisqr": _chisqr / (len(self._x) - len(_result.x)),
            "alpha": _alpha,
            "cov": _cov,
            "success": _result.success,
            "message": _result.message,
        }

    def model_interpolation(self, x=None, params=None, extend=0.2):
        """
        Get the interpolated model predictions for a given set of parameters and an optional range of x values.

        Parameters:
        ----------
        x: array-like, optional
            Independent variable data for which to compute the model predictions. If None, a range of x values will be generated based on the range of the original x data, extended by a specified factor.
        params: array-like, optional
            Model parameters for which to compute the model predictions. If None, the best-fit parameters from the fit will be used.
        extend: float, optional
            Factor by which to extend the range of x values if x is not provided. The range of x values will be extended by this factor on both sides of the original range of x data.

        Returns:
        -------
        x: array-like
            Independent variable data for which the model predictions were computed.
        yhat: array-like
            Model predictions for the given parameters and x values.
        """

        if x is None:
            _x_range = self._x.max() - self._x.min()
            x = np.linspace(
                self._x.min() - extend * _x_range,
                self._x.max() + extend * _x_range,
                1000,
            )

        if params is None:
            params = self.fit()["params"]

        return x, self._get_model(x, params)
