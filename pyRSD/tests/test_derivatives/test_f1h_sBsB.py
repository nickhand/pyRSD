from . import numdifftools, numpy
from pyRSD.rsd.derivatives import compute
    
    
def test_total(driver):
    
    model = driver.theory.model
    
    # get the deriv arguments
    k    = driver.data.combined_k
    mu   = driver.data.combined_mu
    pars = driver.theory.fit_params
    args = (model, pars, k, mu)
    
    # our derivative
    x = compute('f1h_sBsB', *args)
    
    # setup the numerical derivative
    index = driver.theory.free_names.index('f1h_sBsB')
    theta = [driver.theory.fit_params[k].value for k in driver.theory.free_names]
    
    def f(x):
        t = theta.copy()
        t[index] = x
        driver.theory.set_free_parameters(t)
        return driver.theory.model.Pgal(k, mu)
          
    g = numdifftools.Derivative(f, step=1e-3)
    y = g(theta[index])
    
    # compare to 5% accuracy
    numpy.testing.assert_allclose(x, y, rtol=0.05, atol=0.01)
    

    
