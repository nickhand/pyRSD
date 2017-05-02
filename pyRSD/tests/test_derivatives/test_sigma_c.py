from . import numdifftools, numpy as np
import pytest
from pyRSD.rsd.power.gal.derivatives import dPgal_dsigma_c

NMU = 41

@pytest.mark.parametrize("socorr", [True, False])
@pytest.mark.parametrize("fog_model", ['modified_lorentzian', 'lorentzian', 'gaussian'])
def test_partial(driver, socorr, fog_model):

    # set the socorr and fog model
    driver.theory.model.use_so_correction = socorr
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
    x = dPgal_dsigma_c.eval(*args)

    # numerical derivative
    def f(x):
        model.sigma_c = x
        return driver.theory.model.power(k, mu)
    g = numdifftools.Derivative(f, step=1e-3)
    y = g(model.sigma_c)

    # compare
    np.testing.assert_allclose(x, y, rtol=1e-2)
