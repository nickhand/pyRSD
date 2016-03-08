from ... import numpy as np
from ..results import EmceeResults, LBFGSResults
from . import tools

import scipy.optimize
import logging

try:
    from statsmodels.tools import numdiff
except:
    numdiff = None

logger = logging.getLogger('rsdfit.lbfgs_fitter')
logger.addHandler(logging.NullHandler())


def run(params, theory, objective, pool=None, init_values=None):
    """
    Perform nonlinear fitting of a system using `scipy.optimize`.
    
    Any kwargs passed will not be used
    """
    if numdiff is None:
        raise ImportError('`statsmodels` is required to compute derivatives in the scipy.optimize fitter')
        
    if init_values is None:
        raise ValueError("please specify how to initialize the maximum-likelihood solver")
    
    epsilon    = params.get('lbfgs_epsilon', 1e-8)
    factr      = params.get('lbfgs_factr', 1e7)
    use_bounds = params.get('lbfgs_use_bounds', True)
            
    #-----------------
    # do the work
    #-----------------
    
    # compute the gradient using finite difference from statsmodels
    def gradient(theta):
        return numdiff.approx_fprime(theta, objective, epsilon=epsilon)
    
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
    
    exception = False  
    try:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(objective, x0=init_values, bounds=bounds, fprime=gradient, iprint=1)
    except:
        exception = True
        pass
    
    # handle the results
    #---------------------------------------------------------------------------
    # extract the values to put them in the feedback
    if d['warnflag'] != 0:
        logger.error("scipy.optimize: nonlinear fit with method L-BFGS-B failed; %s" %d['task'])
    else:
        logger.info("scipy.optimize: nonlinear fit with method L-BFGS-B succeeded after %d iterations" %d['nit'])
        
    results = None
    if not exception:
        results = LBFGSResults((x, f, d), theory.fit_params)
    return results, exception


        
