from . import numdifftools, numpy as np
import pytest
from pyRSD.rsd.power.gal.derivatives import dPgal_dsigma_so

NMU = 41

@pytest.mark.parametrize("fog_model", ['modified_lorentzian', 'lorentzian', 'gaussian'])
def test_partial(driver, fog_model):

    # set the socorr and fog model
    driver.theory.model.use_so_correction = True
    driver.theory.model.fog_model = fog_model

    model = driver.theory.model

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
    x = dPgal_dsigma_so.eval(*args)

    # numerical derivative
    def f(x):
        model.sigma_so = x
        return driver.theory.model.power(k, mu)
    g = numdifftools.Derivative(f)
    y = g(model.sigma_so)

    # compare
    np.testing.assert_allclose(x, y, rtol=1e-2)
