.. currentmodule:: pyRSD.rsdfit.data

API
---

The main class for handling the power measurements is the :class:`PowerData`
class. To initialize this class, the parameters are:

.. currentmodule:: pyRSD.rsdfit.data.PowerData

.. autosummary::
    covariance
    covariance_Nmocks
    covariance_rescaling
    data_file
    ells
    fitting_range
    grid_file
    max_ellprime
    mode
    mu_bounds
    statistics
    usedata
    window_file

.. currentmodule:: pyRSD.rsdfit.data

.. autoclass:: PowerData
  :members: covariance, covariance_Nmocks, covariance_rescaling, data_file, ells, fitting_range, grid_file, max_ellprime, mode, mu_bounds, statistics, usedata, window_file, to_file, help

Power Statistics
~~~~~~~~~~~~~~~~

.. autoclass:: PowerMeasurements
  :members:

Covariance Matrix
~~~~~~~~~~~~~~~~~

.. autoclass:: PoleCovarianceMatrix
  :members: from_plaintext, to_plaintext, periodic_gaussian_covariance, cutsky_gaussian_covariance

.. autoclass:: PkmuCovarianceMatrix
  :members: from_plaintext, to_plaintext, periodic_gaussian_covariance
