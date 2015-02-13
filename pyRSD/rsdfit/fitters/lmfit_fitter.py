from ... import numpy as np
from ..results import LmfitResults
from . import tools

try:
    import lmfit
except:
    raise ImportError("`lmfit` must be installed to use as a fitter")
import logging
import time

logger = logging.getLogger('rsdfit.lmfit_fitter')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
def run(params, theory, objective, **kwargs):
    """
    Perform nonlinear fitting of a system using `lmfit`.
    
    Any kwargs passed will not be used
    """
    # get params and/or defaults
    method  = params.get('lmfit_method', 'leastsq')

    # make lmfit.Parameters out of all free parameters
    pars = lmfit.Parameters()
    logger.info("LMFIT: Making `Parameter` objects for all free parameters")
    for name in theory.fit_params.free_parameter_names:
        par = theory.fit_params[name]
        minimum, maximum = par.limits
        pars.add(name, value=par.value, min=minimum, max=maximum, vary=True)
        
    #---------------------------------------------------------------------------
    # the objective function, which takes a list of Parameters and returns
    # chi2 in the right format
    def model_eval(pars):
        
        # get names and values
        vals = [pars[par].value for par in pars]
        names = [par for par in pars]
        
        # the objective function takes a numpy array of the values and
        # returns chi2
        toret = objective(np.array(vals))
        chi2 = np.array(toret**2).sum()
        
        # short log message to report to the user
        info = ", ".join(['{}={:.4g}'.format(name, val) for name, val in zip(names, vals)])
        logger.info("LMFIT: current values: {} (chi2={:.6g})".format(info, chi2))

        # return normed residuals
        return toret
    #---------------------------------------------------------------------------
        
    extra_kwargs = {}
    if method == 'leastsq':
        extra_kwargs['epsfcn'] = 1e-3
    
    # run the fitter.
    try:
        logger.info("LMFIT: calling lmfit.minimize with method `{}`".format(method))
        start = time.time()
        result = lmfit.minimize(model_eval, pars, method=method, **extra_kwargs)
        stop = time.time()
        logger.info("LMFIT: ...minimization finished. Time elapsed: {}".format(tools.hms_string(stop-start)))    
    except Exception as msg:
        raise RuntimeError(("Error in running lmfit. Perhaps the version "
            "is not right? (you have {} and should have >{}). Original "
            "error message: {}").format(lmfit.__version__, '0.7', str(msg)))
            
    # handle the results
    #---------------------------------------------------------------------------
    # extract the values to put them in the feedback
    if hasattr(result, 'success') and not result.success:
        logger.error("LMFIT: nonlinear fit with method {} failed".format(method))
    else:
        logger.info("LMFIT: nonlinear fit with method {} succeeded".format(method))
        
    # the same with the errors and correlation coefficients, if there are any
    if not hasattr(result, 'errorbars') or not result.errorbars:
        logger.error("LMFIT: could not estimate errors (set to nan)")
    results = LmfitResults(result)
    
    return results, False
#end run

#-------------------------------------------------------------------------------    
        