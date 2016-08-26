from . import numdifftools, numpy
import pytest
from pyRSD.rsd.derivatives.Pgal import dPgal_dNcBs

@pytest.fixture(scope='module', params=[True, False])
def socorr(request):
    return request.param

def test_dPgal_dNcBs(driver, socorr):
    
    model = driver.theory.model
    
    # set the socorr
    driver.theory.model.use_so_correction = socorr
    
    # get the deriv arguments
    k    = driver.data.combined_k
    mu   = driver.data.combined_mu
    pars = driver.theory.fit_params
    args = (model, pars, k, mu)
    
    # our derivative
    x = dPgal_dNcBs.eval(*args)
    
    # numerical derivative
    def f(x):
        model.NcBs = x
        return driver.theory.model.Pgal(k, mu)
    g = numdifftools.Derivative(f, step=1e-3)
    y = g(model.NcBs)
    
    # compare
    numpy.testing.assert_allclose(x, y, rtol=1e-2)
    

    
