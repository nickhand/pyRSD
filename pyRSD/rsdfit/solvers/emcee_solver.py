from ... import numpy as np, os
from .. import logging
from ..results import EmceeResults
from . import tools, objectives

import time
import signal
import traceback
import functools

logger = logging.getLogger('rsdfit.emcee_fitter')
logger.addHandler(logging.NullHandler())

#------------------------------------------------------------------------------
# context manager for running multiple chains
#------------------------------------------------------------------------------
class ChainManager(object):
    """
    Class to serve as context manager for running multiple chains, which
    will handle exceptions (user-supplied or otherwise) and convergence
    criteria from multiple chains
    """
    def __init__(self, sampler, niters, nwalkers, theory, comm):
        """
        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            the emcee sampler object
        niter : int
            the number of iterations to run
        nwalkers : int
            the number of walkers we are using
        theory : theory.GalaxyPowerParameters
            the set of parameters for the theoretical model
        comm : MPI.Communicator
            the communicator for the the multiple chains
        """
        self.sampler   = sampler
        self.niters    = niters
        self.nwalkers  = nwalkers
        self.theory    = theory
        self.comm      = comm
        self.exception = None
        
        # register the signal handlers and tags
        signal.signal(signal.SIGUSR1, initiate_exit)
        signal.signal(signal.SIGUSR2, initiate_exit)
        signal.signal(signal.SIGQUIT, initiate_exit)
        self.tags = enum('CONVERGED', 'EXIT', 'CTRL_C')
        
        # remember the start time
        self.start    = time.time()
    
    def __enter__(self):
        return self
    
    def update_progress(self, niter):
        conditions = [niter < 10, niter < 50 and niter % 2 == 0,  niter < 500 and niter % 10 == 0, niter % 100 == 0]
        if any(conditions):
            update_progress(self.theory, self.sampler, self.niters, self.nwalkers)
     
    def check_status(self):
        from mpi4py import MPI
        
        if self.comm is not None:
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=self.tags.EXIT):
                raise ExitingException
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=self.tags.CONVERGED):
                raise ConvergenceException
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=self.tags.CTRL_C):
                raise KeyboardInterrupt
    
    def do_convergence(self, niter):
        if niter < 500:
            return False
        elif niter < 1500 and niter % 200 == 0:
            return True
        elif niter >= 1500 and niter % 100 == 0:
            return True
        return False
    
    def check_convergence(self, niter, epsilon, start_iter, start_chain):
        if self.comm is not None and self.do_convergence(start_iter+niter+1):
            chain = self.sampler.chain if start_chain is None else np.concatenate([start_chain, self.sampler.chain],axis=1)
        
            self.comm.Barrier() # sync each chain to same number of iterations
            chains = self.comm.gather(chain, root=0)
            if self.comm.rank == 0:
                converged = test_convergence(chains, start_iter+niter+1, epsilon)
                if converged: raise ConvergenceException
    
    def sample(self, p0, lnprob0):
        kwargs = {}
        kwargs['lnprob0'] = lnprob0
        kwargs['iterations'] = self.niters
        kwargs['storechain'] = True
        return enumerate(self.sampler.sample(p0, **kwargs))
            
    def __exit__(self, exc_type, exc_value, exc_traceback):
        
        # emcee raises a RuntimeError -- check if it was actually a KeyboardInterrupt
        # if isinstance(exc_value, RuntimeError):
        #     tb = traceback.format_exc()
        #     if 'KeyboardInterrupt' in tb:
        #         exc_type = KeyboardInterrupt
        #         exc_value = exc_type()
                
        if isinstance(exc_value, KeyboardInterrupt):
            logger.warning("EMCEE: ctrl+c pressed - saving current state of chain")
            tag = self.tags.CTRL_C
        elif isinstance(exc_value, ConvergenceException):
            logger.warning("EMCEE: convergence criteria satisfied -- exiting")
            tag = self.tags.CONVERGED
        elif exc_value is not None:
            logger.warning("EMCEE: exception occurred - trying to save current state of chain")
            trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback, limit=5))
            logger.warning("   traceback:\n%s" %trace)         
            tag = self.tags.EXIT
        
        exceptions = (ConvergenceException, ExitingException)
        if exc_value is not None and not isinstance(exc_value, exceptions):
            logger.warning("EMCEE: setting exception to true before exiting")
            self.exception = exc_value
            
        # convergence exception
        if exc_value is not None:
            if self.comm is not None:
                for r in range(0, self.comm.size):
                    if r != self.comm.rank: 
                        self.comm.send(None, dest=r, tag=tag)
                            
        # print out some info and exit
        stop = time.time()
        logger.warning("EMCEE: ...iterations finished. Time elapsed: {}".format(tools.hms_string(stop-self.start)))
        logger.warning("EMCEE: mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
        try:
            logger.warning("EMCEE: autocorrelation time: {}".format(self.sampler.get_autocorr_time()))
        except:
            pass

        return True

#------------------------------------------------------------------------------
# tools setup
#------------------------------------------------------------------------------
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

class ConvergenceException(Exception):
    pass

class ExitingException(Exception):
    pass
    
def initiate_exit(signum, stack):
    raise ExitingException
    
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
    text += ["      best logp = {:.6g} (reached at iter {}, walker {})".format(logprobs.max(axis=0)[best_iter], best_iter, best_walker)]
    try:
        acc_frac = sampler.acceptance_fraction
        acor = sampler.acor
    except:
        acc_frac = np.array([np.nan]) 
        acor = np.zeros(len(theory.free))
            
    text += ["      acceptance_fraction ({}->{} (median {}))".format(acc_frac.min(), acc_frac.max(), np.median(acc_frac))]      
    for i, par in enumerate(theory.free):
        pos = chain[:,-last:,i].ravel()
        text.append("  {:15s} = {:.6g} +/- {:<12.6g} (best={:.6g}) (autocorr: {:.3g})".format(par.name,
                                            np.median(pos), np.std(pos), chain[best_walker, best_iter, i], acor[i]))
    text = "\n".join(text) +'\n'
    logger.warning(text)
     
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
    

#------------------------------------------------------------------------------
# the main function to runs
#------------------------------------------------------------------------------
def run(params, theory, pool=None, chains_comm=None, init_values=None):
    """
    Perform MCMC sampling of the parameter space of a system using `emcee`
        
    Parameters
    ----------
    params : ParameterSet
        This holds the parameters needed to run the `emcee` fitter
    theory : GalaxyPowerTheory
        Theory object, which has the `fit_params` ParameterSet as an attribute
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
    import emcee
         
    # get params and/or defaults
    nwalkers  = params.get('walkers', 20)
    niters    = params.get('iterations', 500)
    label     = params.get('label')
    ndim      = theory.ndim
    init_from = params.get('init_from', 'prior')
    epsilon   = params.get('epsilon', 0.02)
    test_conv = params.get('test_convergence', False)
    
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
    if init_from in ['maximum_probability', 'fiducial', 'result']:
        if init_values is None:
            raise ValueError("EMCEE: cannot initialize around best guess -- none provided")
        
        labels = {'maximum_probability' : 'maximum probability', 'fiducial': 'fiducial', 'result': "previous result best-fit"}
        lab = labels[init_from]
        
        # initialize in random ball
        # shape is (nwalkers, ndim)
        p0 = np.array([init_values + 1e-3*np.random.randn(ndim) for i in range(nwalkers)])
        logger.warning("EMCEE: initializing walkers in random ball around %s parameters" %lab)
            
    # 2) initialize and restart from previous run
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
    objective = functools.partial(objectives.lnprob)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective, pool=pool)

    # iterator interface allows us to tap ctrl+c and know where we are
    niters -= start_iter
    burnin = 0 if start_iter > 0 else params.get('burnin', 100)
    logger.warning("EMCEE: running {} iterations with {} free parameters...".format(niters, ndim))

    #---------------------------------------------------------------------------
    # do the sampling
    #---------------------------------------------------------------------------
    with ChainManager(sampler, niters, nwalkers, theory, chains_comm) as manager:
        for niter, result in manager.sample(p0, lnprob0):
            
            # check if we need to exit due to exception/convergence
            manager.check_status()
            
            # update progress and test convergence
            manager.update_progress(niter)
            if test_conv:
                manager.check_convergence(niter, epsilon, start_iter, start_chain)
    
    # make the results and return
    new_results = EmceeResults(sampler, theory.fit_params, burnin)
    if old_results is not None:
        new_results = old_results + new_results
        
    exception_raised = False
    if manager.exception is not None  and not isinstance(manager.exception, KeyboardInterrupt):
        exception_raised = True
    logger.warning("EMCEE: exiting EMCEE fitter with exception = %s" %str(exception_raised))
    return new_results, manager.exception

    
