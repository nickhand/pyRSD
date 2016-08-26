from . import numdifftools, numpy
from pyRSD.rsd.derivatives.Pgal import dPgal_dNsBsB

def test_dPgal_dNsBsB(driver):
    
    model = driver.theory.model
    
    # get the deriv arguments
    k    = driver.data.combined_k
    mu   = driver.data.combined_mu
    pars = driver.theory.fit_params
    args = (model, pars, k, mu)
    
    # our derivative
    x = dPgal_dNsBsB.eval(*args)
    
    # numerical derivative
    def f(x):
        model.NsBsB = x
        return driver.theory.model.Pgal(k, mu)
    g = numdifftools.Derivative(f, step=1e-3)
    y = g(model.NsBsB)
    
    # compare
    numpy.testing.assert_allclose(x, y, rtol=1e-2)
    

    
