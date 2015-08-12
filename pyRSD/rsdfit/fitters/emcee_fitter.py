from ... import numpy as np, os
from .. import logging
from ..results import EmceeResults
from . import tools

import emcee
import time

logger = logging.getLogger('rsdfit.emcee_fitter')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
def update_progress(theory, sampler, niters, nwalkers, last=10):
    """
    Report the current status of the sampler.
    """
    k = sum(sampler.lnprobability[0] < 0)
    if not k:
        logger.warning("No iterations with valid parameters (chain shape = {})".format(sampler.chain.shape))
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
    for i, par in enumerate(theory.free):
        pos = chain[:,-last:,i].ravel()
        text.append("  {:15s} = {:.6g} +/- {:<12.6g} (best={:.6g}) (autocorr: {:.3g})".format(par.name,
                                            np.median(pos), np.std(pos), chain[best_walker, best_iter, i], acor[i]))
    
    text = "\n".join(text) +'\n'
    logger.warning(text)
    
    
def do_convergence(niter):
    """
    Determine if we should check convergence at this iteration
    """
    if niter < 500:
        return False
    elif niter < 1500 and niter % 200 == 0:
        return True
    elif niter >= 1500 and niter % 100 == 0:
        return True

    return False
    
def test_convergence(chains0, niter, epsilon):
    """
    Test convergence using the Gelman-Rubin diagnostic
    
    # Calculate Gelman & Rubin diagnostic
    # 1. Remove the first half of the current chains
    # 2. Calculate the within chain and between chain variances
    # 3. estimate your variance from the within chain and between chain variance
    # 4. Calculate the potential scale reduction parameter
    """
    (walkers, _, ndim) = chains0[0].shape
    Nchains = len(chains0)
    n = max(niter/2, 1)
    
    chains, withinchainvar, meanchain = [],[],[]
    for chain in chains0:
        inds = np.nonzero(chain)
        chain = chain[inds].reshape((walkers, -1, ndim))
        chain = chain[:, n:, :].reshape((-1, ndim))
        chains.append(chain)
        withinchainvar.append(np.var(chain, axis=0))
        meanchain.append(np.mean(chain, axis=0))
    
    meanall = np.mean(meanchain, axis=0)
    W = np.mean(withinchainvar, axis=0)
    B = np.zeros(ndim)

    for jj in range(0, Nchains):
        B += n*(meanall - meanchain[jj])**2 / (Nchains-1.)
    estvar = (1. - 1./n)*W + B/n
    with np.errstate(invalid='ignore'):
        scalereduction = np.sqrt(estvar/W)
    
    converged = abs(1.-scalereduction) <= epsilon
    logger.warning("EMCEE: testing convergence with epsilon = %.4f" %epsilon)
    logger.warning("            %d/%d parameters have converged" %(converged.sum(), ndim))
    logger.warning("            scale-reduction = %s" %str(scalereduction))
    return np.all(converged)
    
#-------------------------------------------------------------------------------
def run(params, theory, objective, pool=None, chains_comm=None, init_values=None):
    """
    Perform MCMC sampling of the parameter space of a system using `emcee`.
        
        
    Parameters
    ----------
    params : ParameterSet
        This holds the parameters needed to run the `emcee` fitter
    theory : GalaxyPowerTheory
        Theory object, which has the `fit_params` ParameterSet as an attribute
    objective : callable
        The function that results the log of the probability
    pool : emcee.MPIPool, optional
        Pool object if we are using MPI to run emcee
    init_values : array_like, `EmceeResults`
        Initial positions; if not `None`, initialize the emcee walkers
        in a small, random ball around these positions
        
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
    init_from = params.get('init_from', 'prior')
    epsilon   = params.get('epsilon', 0.02)
    test_conv = params.get('test_convergence', True)
    
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
    old_results = None
    start_iter = 0
    lnprob0 = None
    start_chain = None
    
    # 1) initialixe from initial provided values
    if init_from == 'max-like' or init_from == 'fiducial':
        if init_values is None:
            raise ValueError("EMCEE: cannot initialize around best guess -- none provided")
        
        # initialize in random ball
        # shape is (nwalkers, ndim)
        p0 = np.array([init_values + 1e-3*np.random.randn(ndim) for i in range(nwalkers)])
        logger.warning("EMCEE: initializing walkers in random ball around best guess parameters")
            
    # 2) initialize from past run
    elif isinstance(init_values, EmceeResults):
        
        # copy the results object
        old_results = init_values.copy()
        
        # get the attributes
        start_chain = old_results.chain
        lnprob0 = old_results.lnprobs[:,-1]
        start_iter = start_chain.shape[1]
        p0 = np.array(start_chain[:, -1, :])
        
        logger.warning("EMCEE: continuing previous run (starting at iteration {})".format(start_iter))
    
    # 3) start from scratch
    else:
        if init_from == 'previous_run':
            raise ValueError('trying to init from previous run, but old chain failed')

        # Initialize a set of parameters
        try:
            logger.warning("Attempting multivariate initialization from {}".format(init_from))
            p0, drew_from = tools.multivariate_init(theory, nwalkers, draw_from=init_from, logger=logger)
            logger.warning("Initialized walkers from {} with multivariate normals".format(drew_from))
        except ValueError:
            logger.warning("Attempting univariate initialization")
            p0, drew_from = tools.univariate_init(theory, nwalkers, draw_from=init_from, logger=logger)
            logger.warning("Initialized walkers from {} with univariate distributions".format(drew_from))
    
    # initialize the sampler
    logger.warning("EMCEE: initializing sampler with {} walkers".format(nwalkers))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective, pool=pool)

    # iterator interface allows us to tap ctrl+c and know where we are
    exception = False
    converged = False
    niters -= start_iter
    burnin = 0 if start_iter > 0 else params.get('burnin', 100)
    try:                           
        logger.warning("EMCEE: running {} iterations with {} free parameters...".format(niters, ndim))
        start = time.time()    
        generator = sampler.sample(p0, lnprob0=lnprob0, iterations=niters, storechain=True)
     
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
                
            # check convergence
            if chains_comm is not None and test_conv and do_convergence(start_iter+niter+1):
                chain = sampler.chain if start_chain is None else np.concatenate([start_chain, sampler.chain],axis=1)
                
                chains_comm.Barrier() # sync each chain to same number of iterations
                chains = chains_comm.gather(chain, root=0)
                if chains_comm.rank == 0:
                    converged = test_convergence(chains, start_iter+niter+1, epsilon)
                chains_comm.Barrier() # sync each chain to same number of iterations
                converged = chains_comm.bcast(converged, root=0)
                if converged: raise Exception

        stop = time.time()
        logger.warning("EMCEE: ...iterations finished. Time elapsed: {}".format(tools.hms_string(stop-start)))

        # acceptance fraction should be between 0.2 and 0.5 (rule of thumb)
        logger.warning("EMCEE: mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        try:
            logger.warning("EMCEE: autocorrelation time: {}".format(sampler.get_autocorr_time()))
        except:
            pass
        
    except KeyboardInterrupt:
        logger.warning("EMCEE: ctrl+c pressed - saving current state of chain")
    except Exception as e:
        if not converged:
            exception = True
            logger.warning("EMCEE: exception occurred - trying to save current state of chain")
            logger.warning("EMCEE: exception message: %s" %e)
        else:
            logger.warning("EMCEE: convergence criteria satisfied -- exiting")
        logger.warning("EMCEE: current parameters:\n %s" %str(theory.fit_params))

    #finally:
    new_results = EmceeResults(sampler, theory.fit_params, burnin)
    if old_results is not None:
        new_results = old_results + new_results
    
    return new_results, exception

#-------------------------------------------------------------------------------
    
