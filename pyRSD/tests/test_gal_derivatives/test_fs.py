from . import numdifftools, numpy as np
import pytest
from pyRSD.rsd.power.gal.derivatives import dPgal_dfs
from pyRSD.rsd.power.gradient import compute
from pyRSD.rsd.power.gal.derivatives import PgalDerivative

NMU = 41

@pytest.mark.parametrize("socorr", [True, False])
def test_partial(driver, socorr):

    # set the socorr
    driver.theory.model.use_so_correction = socorr
    model = driver.theory.model
    driver.set_fiducial()

    # get the deriv arguments
    k    = driver.data.combined_k
    mu = np.linspace(0., 1., NMU)

    # broadcast to the right shape
    k     = k[:, np.newaxis]
    mu    = mu[np.newaxis, :]
    k, mu = np.broadcast_arrays(k, mu)
    k     = k.ravel(order='F')
    mu    = mu.ravel(order='F')

    pars = driver.theory.fit_params
    args = (model, pars, k, mu)

    # our derivative
    x = dPgal_dfs.eval(*args)

    # numerical derivative
    def f(x):
        model.fs = x
        return driver.theory.model.power(k, mu)
    g = numdifftools.Derivative(f, step=1e-3)
    y = g(model.fs)

    # compare
    np.testing.assert_allclose(x, y, rtol=1e-2)

@pytest.mark.parametrize("socorr", [True, False])
def test_total(driver, socorr):

    # set the socorr
    driver.theory.model.use_so_correction = socorr
    model = driver.theory.model
    driver.set_fiducial()

    # get the deriv arguments
    k    = driver.data.combined_k
    mu = np.linspace(0., 1., NMU)

    # broadcast to the right shape
    k    = driver.data.combined_k
    k     = k[:, np.newaxis]
    mu    = mu[np.newaxis, :]
    k, mu = np.broadcast_arrays(k, mu)
    k     = k.ravel(order='F')
    mu    = mu.ravel(order='F')

    pars = driver.theory.fit_params
    args = (model, pars, k, mu)

    # our derivative
    x = compute(PgalDerivative.registry(), 'fs', *args)

    # setup the numerical derivative
    index = driver.theory.free_names.index('fs')
    theta = driver.theory.free_values

    def f(x):
        t = theta.copy()
        t[index] = x
        driver.theory.set_free_parameters(t)
        return driver.theory.model.power(k, mu)

    g = numdifftools.Derivative(f, step=1e-3)
    y = g(theta[index])

    # compare to 5% accuracy
    np.testing.assert_allclose(x, y, rtol=0.05, atol=0.01)
