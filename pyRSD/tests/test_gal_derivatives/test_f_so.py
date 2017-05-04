from . import numdifftools, numpy as np
from pyRSD.rsd.power.gal.derivatives import dPgal_df_so

NMU = 41

def test_partial(driver):

    # set the socorr
    driver.theory.model.use_so_correction = True
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
    x = dPgal_df_so.eval(*args)

    # numerical derivative
    def f(x):
        model.f_so = x
        return driver.theory.model.power(k, mu)
    g = numdifftools.Derivative(f, step=1e-3)
    y = g(model.f_so)

    # compare
    np.testing.assert_allclose(x, y, rtol=1e-2)
