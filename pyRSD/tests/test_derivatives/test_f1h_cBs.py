from . import numdifftools, numpy as np
from pyRSD.rsd.power.gradient import compute
from pyRSD.rsd.power.gal.derivatives import PgalDerivative

NMU = 41

def test_total(driver):

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
    x = compute(PgalDerivative.registry(), 'f1h_cBs', *args)

    # setup the numerical derivative
    index = driver.theory.free_names.index('f1h_cBs')
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
