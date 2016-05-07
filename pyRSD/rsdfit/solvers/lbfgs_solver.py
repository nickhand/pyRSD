from ... import numpy as np
from ..results import LBFGSResults
from .. import logging
from . import tools, objectives, lbfgs

import functools
import time

logger = logging.getLogger('rsdfit.lbfgs_fitter')
    
def lnlike_objective(x=None):
    driver = get_rsdfit_driver()
    return driver.lnlike(x)

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
                       
def run(params, theory, pool=None, init_values=None):
    """
    Perform nonlinear fitting of a system using `scipy.optimize`.
    
    Any kwargs passed will not be used
    """
    init_from = params['init_from'].value
    
    # draw initial values randomly from prior
    if init_from == 'prior':
        init_values = InitializeFromPrior(theory.fit_params.free)
        
    # add some scatter to initial values
    elif init_from in ['fiducial', 'result']:
        scatter = params.get('init_scatter', 0.)
        if scatter > 0:
            init_values = InitializeWithScatter(theory.fit_params.free, init_values, scatter)
                    
    if init_values is None:
        raise ValueError("please specify how to initialize the maximum-likelihood solver")
    
    epsilon    = params.get('lbfgs_epsilon', 1e-4)
    use_priors = params.get('lbfgs_use_priors', True)
    options    = params.get('lbfgs_options', {})
    options['test_convergence'] = params.get('test_convergence', True)
    
    # sort epsilon is a dictionary of values
    if isinstance(epsilon, dict):
        epsilon = np.array([epsilon.get(k, 1e-4) for k in theory.free_names])

    # log some info
    if use_priors:
        logger.info("running LBFGS minimizer with priors")
    else:
        logger.info("running LBFGS minimizer without priors")
            
    # setup the logging header
    names = "   ".join(["%9s" %name for name in theory.free_names])
    logging.info('{0:4s}   {1:s}   {2:9s}'.format('Iter', names, 'F(X)'))
     
    # determine the objective functions
    if use_priors:
        f = functools.partial(objectives.neg_lnprob)
    else:
        f = functools.partial(objectives.neg_lnlike)
    fprime = functools.partial(objectives.gradient, epsilon=epsilon, pool=pool, use_priors=use_priors)
    
    #--------------------------------------------------------------------------
    # run the algorithm, catching any errors
    #--------------------------------------------------------------------------
    exception = False  
    
    # initialize the minimizer
    if isinstance(init_values, LBFGSResults):
        minimizer = lbfgs.LBFGS.from_restart(f, fprime, init_values.data)
        niter = minimizer.data['iteration']
        logger.warning("LBFGS: continuing from previous optimziation (starting at iteration {})".format(niter))
    else:
        minimizer = lbfgs.LBFGS(f, fprime, init_values)
    
    try:
        start = time.time()
        result = minimizer.run_nlopt(**options)
    except KeyboardInterrupt:
        pass
    except:
        import traceback
        logger.warning("exception occured:\n%s" %traceback.format_exc())
        exception = True
        pass
    stop = time.time()
    
    #--------------------------------------------------------------------------
    # handle the results
    #--------------------------------------------------------------------------
    logger.warning("...LBFGS optimization finished. Time elapsed: {}".format(tools.hms_string(stop-start)))
    d = minimizer.data
    
    # extract the values to put them in the feedback
    if exception or d['status'] <= 0:
        reason = minimizer.convergence_status
        msg = "nonlinear fit with method L-BFGS-B failed: %s" %reason
        logger.error(msg)
    else:
        args = (d['iteration'], d['funcalls'])
        logger.info("nonlinear fit with method L-BFGS-B succeeded after %d iterations and %d function evaluations" %args)
        logger.info("   convergence message: %s" %d['status'])
        
    results = None
    if not exception:
        results = LBFGSResults(d, theory.fit_params)
    return results, exception


        
