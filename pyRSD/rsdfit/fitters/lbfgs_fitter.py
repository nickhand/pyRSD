from ... import numpy as np
from ..results import LBFGSResults
from . import tools

import scipy.optimize
import logging
import time

logger = logging.getLogger('rsdfit.lbfgs_fitter')
logger.addHandler(logging.NullHandler())

def InitializeFromPrior(params):
    """
    Initialize by drawing from the prior
    """
    while True:
        p0 = np.array([p.get_value_from_prior(size=1) for p in params])
        for i, value in enumerate(p0):
            params[i].value = value

        if all(p.within_bounds for p in params):
           break

    return p0

def InitializeWithScatter(params, x, scatter):
    """
    Initialize by drawing from the prior
    """
    scale = scatter*x
    
    while True:
        p0 = x + np.random.normal(scale=scale)
        for i, value in enumerate(p0):
            params[i].value = value

        if all(p.within_bounds for p in params):
           break

    return p0
                       
def run(params, theory, objective, pool=None, init_values=None):
    """
    Perform nonlinear fitting of a system using `scipy.optimize`.
    
    Any kwargs passed will not be used
    """
    init_from = params['init_from'].value
    if init_from == 'prior':
        init_values = InitializeFromPrior(theory.fit_params.free)
    elif init_from in ['fiducial', 'result']:
        scatter = params.get('init_scatter', 0.)
        if scatter > 0:
            init_values = InitializeWithScatter(theory.fit_params.free, init_values, scatter)
        
    if init_values is None:
        raise ValueError("please specify how to initialize the maximum-likelihood solver")
    
    epsilon    = params.get('lbfgs_epsilon', 1e-8)
    factr      = params.get('lbfgs_factor', 1e3)
    use_bounds = params.get('lbfgs_use_bounds', False)
    use_priors = params.get('lbfgs_use_priors', True)

    # log some info
    if use_priors:
        logger.info("running LBFGS minimizer with priors")
    else:
        logger.info("running LBFGS minimizer without priors")
    if use_bounds:
        logger.info("running LBFGS minimizer with bounds")
    else:
        logger.info("running LBFGS minimizer without bounds")
    
    # bounds
    bounds = None
    if use_bounds:
        bounds = []
        eps = 1e-5
        for par in theory.fit_params.free:
        
            lower = par.lower
            min_val =  par.min + eps
            if lower is not None:
                min_val = max(min_val, lower)
        
            upper = par.upper
            max_val = par.max - eps
            if upper is not None:
                max_val = min(max_val, upper)
            bounds.append((min_val, max_val))
    
    # call the objective which returns f, fprime
    def _lbfgs_objective(x):
        return objective(x, epsilon=epsilon, pool=pool, use_priors=use_priors)
    
    logger.info("scipy.optimize: LBFGS convergence factor = %.1e" %factr)
    
    # setup the logging header
    names = "   ".join(["%9s" %name for name in theory.free_names])
    logging.info('{0:4s}   {1:s}   {2:9s}'.format('Iter', names, 'f(X)'))
     
    exception = False  
    try:
        start = time.time()
        x, f, d = scipy.optimize.fmin_l_bfgs_b(_lbfgs_objective, m=100, factr=factr, x0=init_values, bounds=bounds)
        
    except:
        import traceback
        logger.warning("exception occured:\n%s" %traceback.format_exc())
        exception = True
        pass
    stop = time.time()
    
    # handle the results
    #---------------------------------------------------------------------------
    logger.warning("scipy.optimize: ...optimization finished. Time elapsed: {}".format(tools.hms_string(stop-start)))
    
    # extract the values to put them in the feedback
    if exception or d['warnflag'] != 0:
        msg = "scipy.optimize: nonlinear fit with method L-BFGS-B failed"
        if not exception: msg += "; %s" %d['task']
        logger.error(msg)
    else:
        logger.info("scipy.optimize: nonlinear fit with method L-BFGS-B succeeded after %d iterations" %d['nit'])
        logger.info("   convergence message: %s" %d['task'])
        
    results = None
    if not exception:
        results = LBFGSResults((x, f, d), theory.fit_params)
    return results, exception


        
