from ... import numpy as np
from ..results import EmceeResults, LBFGSResults
from . import tools

import scipy.optimize
import logging
import functools

logger = logging.getLogger('rsdfit.lbfgs_fitter')
logger.addHandler(logging.NullHandler())
                       
def add_epsilon(theta, i, eps):
    toret = theta.copy()
    toret[i] += eps
    return toret

def lbfgs_objective(x0, f, epsilon=1e-8, pool=None):
    """
    Return the function we are optimizing at `x0` plus its derivative, 
    which is the forward finite-difference gradient
    """
    N = len(x0)
    f0 = f(x0) # value of function at x0

    x = [add_epsilon(x0, i, epsilon) for i in range(N)]
    if pool is None:
        M = map
    else:
        M = pool.map
    
    derivs = np.array(M(f, x))
    return f0, (derivs - f0) / epsilon

def run(params, theory, objective, pool=None, init_values=None):
    """
    Perform nonlinear fitting of a system using `scipy.optimize`.
    
    Any kwargs passed will not be used
    """
    if init_values is None:
        raise ValueError("please specify how to initialize the maximum-likelihood solver")
    
    epsilon    = params.get('lbfgs_epsilon', 1e-8)
    factr      = params.get('lbfgs_factr', 1e7)
    use_bounds = params.get('lbfgs_use_bounds', False)
            
    #-----------------
    # do the work
    #-----------------

    # bounds
    bounds = None
    if use_bounds:
        bounds = []
        for par in theory.fit_params.free:
        
            lower = par.lower
            min_val =  par.min
            if lower is not None:
                min_val = max(min_val, lower)
        
            upper = par.upper
            max_val = par.max
            if upper is not None:
                max_val = min(max_val, upper)
            bounds.append((min_val, max_val))
    
    # call the objective which returns f, fprime
    def _lbfgs_objective(x):
        return lbfgs_objective(x, objective, epsilon=epsilon, pool=pool)
    
    exception = False  
    try:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(_lbfgs_objective, x0=init_values, bounds=bounds, iprint=1)
    except:
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


        
