from . import numdifftools, numpy
from pyRSD.rsd.derivatives.Pgal import dPgal_df_so
    
def test_partial(driver):
    
    # set the socorr
    driver.theory.model.use_so_correction = True
    model = driver.theory.model
    
    # get the deriv arguments
    k    = driver.data.combined_k
    mu   = driver.data.combined_mu
    pars = driver.theory.fit_params
    args = (model, pars, k, mu)
    
    # our derivative
    x = dPgal_df_so.eval(*args)
    
    # numerical derivative
    def f(x):
        model.f_so = x
        return driver.theory.model.Pgal(k, mu)
    g = numdifftools.Derivative(f, step=1e-3)
    y = g(model.f_so)
    
    # compare
    numpy.testing.assert_allclose(x, y, rtol=1e-2)


    
