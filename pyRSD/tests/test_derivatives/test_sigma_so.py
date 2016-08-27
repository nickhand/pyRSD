from . import numdifftools, numpy
import pytest
from pyRSD.rsd.derivatives.Pgal import dPgal_dsigma_so

@pytest.fixture(scope='module', params=['modified_lorentzian', 'lorentzian', 'gaussian'])
def fog_model(request):
    return request.param
     
def test_partial(driver, fog_model):
    
    # set the socorr and fog model
    driver.theory.model.use_so_correction = True
    driver.theory.model.fog_model = fog_model
    
    model = driver.theory.model
    
    # get the deriv arguments
    k    = driver.data.combined_k
    mu   = driver.data.combined_mu
    pars = driver.theory.fit_params
    args = (model, pars, k, mu)
    
    # our derivative
    x = dPgal_dsigma_so.eval(*args)
    
    # numerical derivative
    def f(x):
        model.sigma_so = x
        return driver.theory.model.Pgal(k, mu)
    g = numdifftools.Derivative(f)
    y = g(model.sigma_so)
    
    # compare
    numpy.testing.assert_allclose(x, y, rtol=1e-2)


    
