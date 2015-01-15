from ... import numpy as np
from ..results import EmceeResults

import emcee
import logging
import time
import scipy.stats

logger = logging.getLogger('pyRSD.analysis.emcee_fitter')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

#-------------------------------------------------------------------------------
def univariate_init(theory, nwalkers, draw_from='prior'):
    """
    Initializae variables from univariate prior
    
    Parameters
    ----------
    theory : pyRSD.analysis.theory.GalaxyPowerTheory
        galaxy power theory object, which holds the free parameters for the
        fitting procedure
    nwalkers : int
        the number of walkers used in fitting with `emcee`
    draw_from : str, {'prior', 'posterior'}
        either draw from the prior or the posterior to initialize
    """
    # get the free params
    pars = theory.fit_params.free_parameters
    
    # draw function: if it's from posteriors and a par has no posterior, fall
    # back to prior
    if draw_from == 'posterior':
        draw_funcs = ['get_value_from_posterior' if par.has_posterior() else 'get_value_from_prior' for par in pars]
        get_funcs = ['get_posterior' if par.has_posterior() else 'get_prior' for par in pars]
    else:
        draw_funcs = ['get_value_from_prior' for par in pars]
        get_funcs = ['get_prior' for par in pars]
    
    # create an initial set of parameters from the priors (shape: nwalkers x npar)
    p0 = np.array([getattr(par, draw_func)(size=nwalkers) for par, draw_func in zip(pars, draw_funcs)]).T
    
    # we do need to check if all the combinations produce realistic models
    exceed_max_try = 0
    difficult_try = 0
    
    # loop over each parameter
    for i, walker in enumerate(p0):
        max_try = 100
        current_try = 0
        
        # check the model for this set of parameters
        while True:
            
            # Set the values
            for par, value in zip(pars, walker):
                par.value = value
            
            # if it checks out, continue checking the next one
            if theory.check() or current_try > max_try:
                if current_try > max_try:
                    exceed_max_try += 1
                elif current_try > 50:
                    difficult_try += 1
                p0[i] = walker
                break
            
            current_try += 1
            
            # else draw a new value: for traces, we remember the index so that
            # we can take the parameters from the same representation always
            walker = []
            for ii, par in enumerate(pars):
                value = getattr(par, draw_funcs[ii])(size=1)                    
                walker.append(value)
    
    # Perhaps it was difficult to initialise walkers, warn the user
    if exceed_max_try or difficult_try:
        logger.warning(("Out {} walkers, {} were difficult to initialise, and "
                        "{} were impossible: probably your priors are very "
                        "wide and allow many unphysical combinations of "
                        "parameters.").format(len(p0), difficult_try, exceed_max_try))
    
    # report what was used to draw from:
    if np.all(np.array(get_funcs)=='get_prior'):
        drew_from = 'prior'
    elif np.all(np.array(get_funcs)=='get_posterior'):
        drew_from = 'posterior'
    else:
        drew_from = 'mixture of priors and posteriors'
    
    return p0, drew_from
#end univariate_init

#-------------------------------------------------------------------------------
def multivariate_init(theory, nwalkers, draw_from='prior'):
    """
    Initialize parameters with multivariate normals

    Parameters
    ----------
    theory : pyRSD.analysis.theory.GalaxyPowerTheory
        galaxy power theory object, which holds the free parameters for the
        fitting procedure
    nwalkers : int
        the number of walkers used in fitting with `emcee`
    draw_from : str, {'prior', 'posterior'}
        either draw from the prior or the posterior to initialize
    """
    # get the free params
    pars = theory.fit_params.free_parameters
    npars = len(pars)
    
    # draw function
    draw_func = 'get_value_from_' + draw_from
    
    # getter
    get_func = draw_from
    
    # check if distributions are traces, otherwise we can't generate
    # multivariate distributions
    for par in pars:
        this_dist = getattr(par, get_func)
        if this_dist is None:
            raise ValueError(("No {} defined for parameter {}, cannot "
                              "initialise "
                              "multivariately").format(draw_from, par.name))
        if not this_dist.name == 'trace':
            raise ValueError(("Only trace distributions can be used to "
                              "generate multivariate walkers ({} "
                              "distribution given for parameter "
                              "{})").format(this_dist, par.name))
    
    # extract averages and sigmas
    averages = [getattr(par, get_func).loc for par in pars]
    sigmas = [getattr(par, get_func).scale for par in pars]
    
    # Set correlation coefficients
    cor = np.zeros((npars, npars))
    for i, ipar in enumerate(pars):
        for j, jpar in enumerate(pars):
            prs = scipy.stats.pearsonr(getattr(ipar, get_func).trace,
                                        getattr(jpar, get_func)().trace)[0]
            cor[i, j] = prs * sigmas[i] * sigmas[j]

    # sample is shape nwalkers x npars
    sample = np.random.multivariate_normal(averages, cor, nwalkers)
    
    # Check if all initial values satisfy the limits and priors. If not,
    # draw a new random sample and check again. Don't try more than 100 times,
    # after that we just need to live with walkers that have zero probability
    # to start with...
    for i, walker in enumerate(sample):
        max_try = 100
        current_try = 0
        while True:
            # adopt the values in the system
            for par, value in zip(pars, walker):
                par.value = value
                sample[i] = walker
            # perform the check; if it doesn't work out, retry
            if not theory.check() and current_try < max_try:
                walker = np.random.multivariate_normal(averages, cor, 1)[0]
                current_try += 1
            else:
                break
        else:
            logger.warning("Walker {} could not be initalised with valid parameters".format(i))
    
    return sample, draw_from
#end multivariate_init


#-------------------------------------------------------------------------------
def update_progress(theory, sampler, niters, nwalkers, last=10):
    """
    Report the current status of the sampler.
    """
    k = sum(sampler.lnprobability[0] < 0)
    if not k:
        logger.warning("No valid models available (chain shape = {})".format(sampler.chain.shape))
        return None
        
    chain = sampler.chain[:,:k,:]
    logprobs = sampler.lnprobability[:,:k]
    best_iter = np.argmax(logprobs.max(axis=0))
    best_walker = np.argmax(logprobs[:,best_iter])
    text = ["EMCEE Iteration {:>6d}/{:<6d}: {} walkers, {} parameters".format(k-1, niters, nwalkers, chain.shape[-1])]
    text+= ["      best logp = {:.6g} (reached at iter {}, walker {})".format(logprobs.max(axis=0)[best_iter], best_iter, best_walker)]
    try:
        acc_frac = sampler.acceptance_fraction
        acor = sampler.acor
    except:
        acc_frac = np.array([np.nan])        
    text += ["      acceptance_fraction ({}->{} (median {}))".format(acc_frac.min(), acc_frac.max(), np.median(acc_frac))]
    for i, par in enumerate(theory.free_parameters):
        pos = chain[:,-last:,i].ravel()
        text.append("  {:15s} = {:.6g} +/- {:.6g} (best={:.6g})".format(par.name,
                                            np.median(pos), np.std(pos), chain[best_walker, best_iter, i]))
    
    text = "\n".join(text) +'\n'
    logger.warning(text)

#-------------------------------------------------------------------------------
def run(params, theory, objective, pool=None):
    """
    Perform MCMC sampling of the parameter space of a system using `emcee`.
        
    Notes
    -----
    
    *   Have a look at the `acortime`, it is good practice to let the sampler
        run at least 10 times the `acortime`. If ``acortime = np.nan``, you
        probably need to take more iterations!
    *   Use as many walkers as possible (hundreds for a handful of parameters)
        Number of walkers must be even and at least twice the number of 
        parameters
    *   Beware of a burn-in period. The most conservative you can get is making
        a histogram of only the final states of all walkers.
    """        
    # get params and/or defaults
    nwalkers  = params.get('walkers', 20)
    niters    = params.get('iterations', 500)
    label     = params.get('label')
    ndim      = theory.ndim
    init_from = params.get('init_from', 'posterior')
    
    #---------------------------------------------------------------------------
    # let's check a few things so we dont mess up too badly
    #---------------------------------------------------------------------------
    
    # now, if the number of walkers is smaller then twice the number of
    # parameters, adjust that number to the required minimum and raise a warning
    if 2*ndim > nwalkers:
        logger.warning("EMCEE: number of walkers ({}) cannot be smaller than 2 x npars: set to {}".format(nwalkers, 2*ndim))
        nwalkers = 2*ndim
    
    if nwalkers % 2 != 0:
        nwalkers += 1
        logger.warning("EMCEE: number of walkers must be even: set to {}".format(nwalkers))
    
    
    #---------------------------------------------------------------------------
    # initialize the parameters
    #---------------------------------------------------------------------------

    # if previous_run was requested, if we end up here it was not possible.
    # therefore we set to start from posteriors
    if init_from == 'previous_run':
        logger.warning("Cannot continue from previous run, falling back to start from posteriors")
        init_from = 'posterior'

    # Initialize a set of parameters
    try:
        logger.warning("Attempting multivariate initialization from {}".format(init_from))
        p0, drew_from = multivariate_init(theory, nwalkers, draw_from=init_from)
        logger.warning("Initialized walkers from {} with multivariate normals".format(drew_from))
    except ValueError:
        logger.warning("Attempting univariate initialization")
        p0, drew_from = univariate_init(theory, nwalkers, draw_from=init_from)
        logger.warning("Initialized walkers from {} with univariate distributions".format(drew_from))
    
    # initialize the sampler
    logger.warning("EMCEE: initializing sampler with {} walkers".format(nwalkers))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective, pool=pool)

    # and run!
    try:                               
        logger.warning("EMCEE: starting {} iterations with {} free parameters...".format(niters, ndim))
        logger.warning("EMCEE: starting positions: {}".format(p0))
        start = time.time()    
        generator = sampler.sample(p0, iterations=niters, storechain=True)
        
        # loop over all the steps
        for niter, result in enumerate(generator):                    
            if niter < 10:
                update_progress(theory, sampler, niters, nwalkers)
            elif niter < 50 and niter % 2 == 0:
                update_progress(theory, sampler, niters, nwalkers)
            elif niter < 500 and niter % 10 == 0:
                update_progress(theory, sampler, niters, nwalkers)
            elif niter % 100 == 0:
                update_progress(theory, sampler, niters, nwalkers)

        stop = time.time()
        logger.warning("EMCEE: ...iterations finished. Time elapsed: {}".format(hms_string(stop-start)))

        # close the pool processes
        if pool is not None: 
            pool.close()

        # acceptance fraction should be between 0.2 and 0.5 (rule of thumb)
        logger.warning("EMCEE: mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        logger.warning("EMCEE: autocorrelation time: {}".format(sampler.get_autocorr_time()))
        
    except:
        pass
    finally:
        results = EmceeResults(sampler, theory.fit_params, params.get('burnin', None))
        
    return results
#end run
#-------------------------------------------------------------------------------
    
