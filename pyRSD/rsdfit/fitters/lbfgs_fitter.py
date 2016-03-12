from ... import numpy as np
from ..results import EmceeResults, LBFGSResults
from . import tools

import scipy.optimize
import logging
import functools

logger = logging.getLogger('rsdfit.lbfgs_fitter')
logger.addHandler(logging.NullHandler())
                       
def run(params, theory, objective, pool=None, init_values=None):
    """
    Perform nonlinear fitting of a system using `scipy.optimize`.
    
    Any kwargs passed will not be used
    """
    if init_values is None:
        raise ValueError("please specify how to initialize the maximum-likelihood solver")
    
    epsilon    = params.get('lbfgs_epsilon', 1e-8)
    factr      = params.get('lbfgs_factr', 1e8)
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
    
    exception = False  
    try:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(_lbfgs_objective, m=1000, x0=init_values, bounds=bounds, iprint=1)
    except:
        import traceback
        logger.warning("exception occured:\n%s" %traceback.format_exc())
        exception = True
        pass
    
    # handle the results
    #---------------------------------------------------------------------------
    # extract the values to put them in the feedback
    if exception or d['warnflag'] != 0:
        msg = "scipy.optimize: nonlinear fit with method L-BFGS-B failed"
        if not exception: msg += "; %s" %d['task']
        logger.error(msg)
    else:
        logger.info("scipy.optimize: nonlinear fit with method L-BFGS-B succeeded after %d iterations" %d['nit'])
        
    results = None
    if not exception:
        results = LBFGSResults((x, f, d), theory.fit_params)
    return results, exception


        
