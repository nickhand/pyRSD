"""
    fitting_driver.py
    pyRSD.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : the driver for the main rsdfit fitter
"""
from .. import numpy as np, os
from . import logging, params_filename, model_filename
from .parameters import ParameterSet
from .theory import GalaxyPowerTheory
from .data import PowerData
from .fitters import *
from .util import rsd_io
from .results import EmceeResults, LBFGSResults
from ..rsd import GalaxySpectrum
import functools
import copy

logger = logging.getLogger('rsdfit.fitting_driver')
logger.addHandler(logging.NullHandler())

def add_epsilon(theta, i, eps):
    toret = theta.copy()
    toret[i] += eps
    return toret

class FittingDriver(object):
    """
    A class to handle the data analysis pipeline, merging together a model, 
    theory, and fitting algorithm
    """
    __metaclass__ = rsd_io.PickeableClass
    
    def __init__(self,
                    param_file, 
                    extra_param_file=None, 
                    pool=None, 
                    chains_comm=None, 
                    init_model=True):
        """
        Initialize the driver with the specified parameters
        
        Parameters
        ----------
        param_file : str
            a string specifying the name of the main parameter file
        extra_param_file : str, optional
            a string specifying the name of a file holding extra extra theory parameters
        pool : mpi4py.Pool, optional
            if provided, a pool of MPI processes to use for ``emcee`` fitting; default is ``None``
        chains_comm : mpi4py.Intracomm
            if provided, an MPI Intracommunicator used to communicate between multiple chains
            running concurrently
        init_model : bool, optional
            if `True`, initialize the theoretical model upon initialization; default is `True`
        """        
        # initialize the data too
        self.data = PowerData(param_file)
        self.mode = self.data.mode # mode is either pkmu/poles
        
        # initialize the theory
        kwargs = {}
        kwargs['extra_param_file'] = extra_param_file
        kwargs['kmin'] = self.data.global_kmin
        kwargs['kmax'] = self.data.global_kmax
        self.theory = GalaxyPowerTheory(param_file, **kwargs)
        
        # generic params
        self.params = ParameterSet.from_file(param_file, tags='driver')
        self.pool = pool
        self.chains_comm = chains_comm
        
        # setup the model for data
        if init_model: self._setup_for_data()
        
        # results are None for now
        self.results = None
        
    #---------------------------------------------------------------------------
    # class methods to start from directory
    #---------------------------------------------------------------------------
    @classmethod
    def from_directory(cls, dirname, results_file=None, model_file=None, init_model=True, **kwargs):
        """
        Load a ``FittingDriver`` from a results directory, reading the 
        ``params.dat`` file, and optionally loading a pickled model and 
        a results object
        
        Parameters
        ----------
        dirname : str
            the name of the directory holding the results
        results_file : str, optional
            the name of the file holding the results. Default is ``None``
        model_file : str, optional
            the name of the file holding the model to load. Default is ``None``
        init_model : bool, optional
            whether to initialize the RSD model upon loading. If a model
            file exists in the specified directory, the model is loaded and
            no new model is initialized
        """
        if not os.path.isdir(dirname):
            raise rsd_io.ConfigurationError('`%s` is not a valid directory' %dirname)
        params_path = os.path.join(dirname, params_filename)
        
        model_path = model_file
        existing_model = isinstance(model_path, str) and os.path.exists(model_path)
        existing_model = existing_model or isinstance(model_path, GalaxySpectrum)
        if model_path is not None:
            if not existing_model:
                raise rsd_io.ConfigurationError('provided model file `%s` does not exist' %model_path)
        else:
            model_path = os.path.join(dirname, model_filename)
        if not os.path.exists(params_path):
            raise rsd_io.ConfigurationError('parameter file `%s` must exist to load driver' %params_path)
            
        init_model = (not existing_model and init_model)
        driver = cls(params_path, init_model=init_model, **kwargs)
        
        # set the model and results
        if existing_model:
            driver.model = model_path
        if results_file is not None:
            if not os.path.exists(results_file):
                if not os.path.join(dirname, results_file):
                    raise rsd_io.ConfigurationError('specified results file `%s` does not exist' %results_file)
                else:
                    results_file = os.path.join(dirname, results_file)
            try:
                driver.results = EmceeResults.from_npz(results_file)
            except:
                driver.results = LBFGSResults.from_npz(results_file)
    
        return driver
        
    @classmethod
    def from_restart(cls, dirname, restart_file, iterations, **kwargs):
        """
        Restart chain from a previous run, returning the driver with the results 
        loaded from the specified chain file
        
        Parameters
        ----------
        dirname : str
            the name of the directory holding the results
        restart_file : str
            the name of the file holding the results from the chain to restart from
        iterations : int
            the number of additional iterations to run
        """
        # get the driver
        driver = cls.from_directory(dirname, results_file=restart_file, **kwargs)
        
        # tell driver we are starting from previous run
        driver.params['init_from'].value = 'previous_run'
                
        # set the number of iterations to the total sum we want to do
        driver.params.add('iterations', value=iterations+driver.results.iterations)
        driver.params.add('walkers', value=driver.results.walkers)
        
        # make sure we want to use emcee
        solver_name = driver.params.get('fitter', 'emcee').lower()
        if solver_name != 'emcee':
            raise ValueError("cannot restart chain if desired solver is not `emcee`")
        
        return driver
    
    def to_file(self, filename, mode='w'):
        """
        Write the parameters of this driver to a file
        
        Parameters
        ----------
        filename : str
            the name of the file to write out
        mode : {'a', 'w'}
            the mode to use when writing to file
        """
        kwargs = {'mode':mode, 'header_name':'driver_params', 'footer':True, 'as_dict':False}
        self.params.to_file(filename, **kwargs)
        self.data.to_file(filename, mode='a')
        self.theory.to_file(filename, mode='a')
            
    def run(self):
        """
        Run the whole fitting analysis, from start to finish
        """
        init_from = self.params.get('init_from', None)
        init_values = None
        
        # check for deprecated init_values
        if init_from == 'max-like':
            raise ValueError("``init_from = max-like`` has been deprecated; use `maximum_probability` instead")
        if init_from == 'chain':
            raise ValueError("``init_from = chain`` has been deprecated; use `result` instead")
        
        # init from maximum probability solution
        if init_from == 'maximum_probability':
            init_values = self.find_peak_probability()
            
        # init from fiducial values
        elif init_from == 'fiducial':
            init_values = self.theory.free_fiducial
                
        # init from previous result
        elif init_from == 'result':
            
            # the result to start from
            start_from = self.params.get('start_from', None)
            if start_from is None:
                raise ValueError("if ``init_from = 'result'``, a `start_from` file path must be provided")
            
            # load the result and get the best-fit values
            try:
                result = EmceeResults.from_npz(start_from)
                best_values = result.max_lnprob_values()
            except:
                result = LBFGSResults.from_npz(start_from)
                best_values = result.min_chi2_values
            best_values = dict(zip(result.free_names, best_values))

            init_values = np.empty(self.Np)
            for i, key in enumerate(self.theory.free_names):
                if key in best_values:
                    init_values[i] = best_values[key]
                else:
                    if self.theory.fit_params[key].has_fiducial is not None:
                        init_values[i] = self.theory.fit_params[key].fiducial
                    else:
                        raise ValueError("cannot initiate from previous result -- `%s` parameter missing" %key)
            
        
        # get the solver function
        kwargs = {}
        solver_name = self.params.get('fitter', 'emcee').lower()
        
        # emcee
        if solver_name == 'emcee':
            
            # do not use any analytic approximations for bounds and priors
            # since MCMC can handle discontinuity in log-prob
            for name in self.theory.free_names:
                self.theory.fit_params[name].analytic = False
                
            solver = emcee_fitter.run
            objective = functools.partial(FittingDriver.lnprob, self)
            
            # add some kwargs to pass too
            kwargs['pool'] = self.pool
            kwargs['chains_comm'] = self.chains_comm
            if init_from in ['max-like', 'fiducial', 'chain']: 
                kwargs['init_values'] = init_values
            elif init_from == 'previous_run':
                kwargs['init_values'] = self.results.copy()
                
        # lbfgs
        elif solver_name == 'lbfgs':
            solver = lbfgs_fitter.run
                        
            # use analytic approximation for bounds and priors
            # to enforce continuity
            for name in self.theory.free_names:
                self.theory.fit_params[name].analytic = True
                
            # explictly pass the lnlike function to avoid re-pickling it in parallel
            lnlike_objective = functools.partial(FittingDriver.lnlike, self)
            objective = functools.partial(FittingDriver.minimize_objective, self, lnlike_objective)
            kwargs = {'init_values': init_values, 'pool':self.pool}
            
        # incorrect
        else:
            raise NotImplementedError("fitter with name '{}' not currently available".format(solver_name))
          
        # run the solver and store the results
        if init_from == 'previous_run':
            logger.info("Restarting `emcee` solver from an old chain")
        else:
            logger.info("Calling the '{}' fitter's solve function".format(solver_name))
        self.results, exception = solver(self.params, copy.deepcopy(self.theory), objective, **kwargs)
        logger.info("...fitting complete")
        
        return exception
    
    def minimize_objective(self, lnlike, theta=None, epsilon=1e-8, pool=None, use_priors=True):
        """
        The objective function for minimizing the negative log probability
        
        Notes
        -----
        This uses either the maximum likelihood (ML) (excluding log priors) or 
        maximum a posteriori probability (MAP) (including log priors) estimation
        
        Parameters
        ----------
        theta : array_like, optional
            if provided, the values of the free parameters to compute the probability
            at; it not provided, use the current values of free parameters from 
            `theory.fit_params`
        lnlike : 
        epsilon : float, optional
            the step-size to use in the finite-difference derivative calculation; 
            default is 1e-8
        pool : MPIPool, optional
            a MPI Pool object to distribute the calculations of derivatives to 
            multiple processes in parallel
        use_priors : bool, optional
            whether to include the log priors in the objective function when 
            minimizing the negative log probability
        """            
        if use_priors:
            nlp0 = -self.lnprior()
            ndlp = -self.theory.dlnprior
        
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)
            
            # if parameters are out of bounds, return the log-likelihood
            # for a null model + the prior results (which hopefully are restrictive)
            if not in_bounds:
                return -self.null_lnlike - self.lnprior(), -self.theory.dlnprior
        else:
            theta = self.theory.free_values

        # value at theta
        nll0 = -lnlike()
        
        # derivatives
        x = [add_epsilon(theta, i, epsilon) for i in range(self.Np)]
        
        if pool is None:
            M = map
        else:
            M = pool.map
    
        # compute the derivatives, optionally in parallel
        derivs = -1.*np.array(M(lnlike, x)) # negative likelihood
        derivs = (derivs - nll0) / epsilon
        prob = nll0
        
        # add in log prior prob and derivatives of log priors
        if use_priors: 
            derivs += ndlp
            prob += nlp0
            
        return prob, derivs
        
    def finalize_fit(self, exception, results_file):
        """
        Finalize the fit, saving the results file
        """                
        # save the results as a pickle
        logger.info('Saving the results to `%s`' %results_file)
        self.results.to_npz(results_file)
        
        if not exception:
            self.results.summarize_fit()

    def find_peak_probability(self):
        """
        Find the peak of the probability distribution
        
        This uses either the maximum likelihood (ML) (excluding log priors) or 
        maximum a posteriori probability (MAP) (including log priors) estimation
        """
        # use analytic approximation for bounds and priors
        # to enforce continuity
        for name in self.theory.free_names:
            self.theory.fit_params[name].analytic = True
            
        # get the solver and objective
        solver = bfgs_fitter.run
        objective = functools.partial(FittingDriver.minimize_objective, self)
        
        # init values from fiducial
        init_values = self.theory.free_fiducial
        
        logger.info("using L-BFGS soler to find the maximum probability values to use as initialization")
        results, exception = solver(params, copy.deepcopy(self.theory), objective, pool=self.pool, init_values=init_values)
        logger.info("...done compute maximum probability")
        
        values = results.min_chi2_values
        del results
        return values

    #---------------------------------------------------------------------------
    # setup functions
    #---------------------------------------------------------------------------
    def _setup_for_data(self):
        """
        Setup the model callables for this set of data
        """
        # initialize the model
        logger.info("Initializing theoretical model")
        self.theory.update_model()
        self.theory.model.initialize()
        logger.info("...theoretical model initialized")
        
        # the model callable
        self._model_callable = self.theory.model_callable(self.data)
                                        
    @property
    def fiducial(self):
        """
        Convenience function to return the fiducial values 
        """
        
    @property
    def results(self):
        """
        The `results` object storing the fitting results
        """
        try:
            return self._results
        except AttributeError:
            return None
        
    @results.setter
    def results(self, val):
        """
        Set the results, checking to make sure we re-order the fitted params
        into the right order
        """
        # possibly reorder the results
        if hasattr(val, 'verify_param_ordering'):
            free_params = self.theory.free_names
            constrained_params = self.theory.constrained_names
            val.verify_param_ordering(free_params, constrained_params)            
        self._results = val
        
    @property
    def model(self):
        """
        The `model` object which returns the P(k,mu) or multipoles theory
        """
        return self.theory.model
        
    @model.setter
    def model(self, val):
        """
        Set the theoretical model
        """
        # set it
        if isinstance(val, basestring):
            logger.info("setting the theoretical model from file `%s`" %val)
        else:
            logger.info("setting the theoretical model from existing instance")
        self.theory.set_model(val)
        
        # print out the model paramete
        params = self.theory.model.to_dict()
        msg = "running with model parameters:\n\n"
        msg += "\n".join(["%-25s: %s" %(k, str(v)) for k,v in sorted(params.iteritems())])
        logger.info(msg)
        
        # model callable
        self._model_callable = self.theory.model_callable(self.data)
        logger.info("...theoretical model successfully read")
        
    @property
    def combined_model(self):
        """
        The model values for each measurement, flattened column-wise
        into a single array
        
        Notes
        -----
        *   the model callable should already returned the flattened
            `combined` values
        """
        return self._model_callable()
        
    @property
    def null_lnlike(self):
        """
        Return the log-likelihood value for a null model, to be used
        when the model parameters are in an invalid state
        """
        try:
            return self._null_lnlike
        except:
            d = self.data.combined_power
            self._null_lnlike = -0.5 * np.dot(d, np.dot(self.data.covariance.inverse, d))
            return self._null_lnlike
            
    @property
    def dof(self):
        """
        Return the degrees of freedom, equal to number of data points minus
        the number of free parameters
        """
        return self.Nb - self.Np
    
    @property
    def Nb(self):
        """
        Return number of data points
        """
        return len(self.combined_model) 
        
    @property
    def Np(self):
        """
        Return the number of free parameters
        """
        return self.theory.ndim
                
    #---------------------------------------------------------------------------
    # probability functions
    #---------------------------------------------------------------------------
    def lnprior(self):
        """
        Return the log of the prior, based on the current values of the free 
        parameters in  `GalaxyPowerTheory`
        """
        return self.theory.lnprior

    def chi2(self, theta=None):
        """
        The chi-squared for the specified model function, based 
        on the current values of the free parameters in `GalaxyPowerTheory`
        
        This returns
        
        ..math: (model - data)^T C^{-1} (model - data)
        """  
        # set the free parameters
        if theta is not None:
            self.theory.set_free_parameters(theta)
              
        diff = self.combined_model - self.data.combined_power
        return np.dot(diff, np.dot(self.data.covariance.inverse, diff))

    def reduced_chi2(self):
        """
        The reduced chi squared value
        """
        return self.chi2() / self.dof
        
    def lnlike(self, theta=None):
        """
        The log of the likelihood, equal to -0.5 * chi2
        """
        return -0.5*self.chi2(theta=theta)
        
    def lnprob(self, theta=None):
        """
        Set the theory free parameters, update the model, and return the log of
        the posterior probability function (to within a constant), defined 
        as likelihood * prior.
        
        This returns:
        
        ..math: -0.5 * self.chi2 + self.lnprior
        """
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)
            
            # return -np.inf if parameters are out of bounds
            if not in_bounds:
                return -np.inf
                
        # check the prior
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        # only compute lnlike if we have finite prior
        else:
            try:
                lnlike = self.lnlike() 
                if np.isnan(lnlike):
                    raise ValueError("log-likelihood calculation resulted in NaN")
            except:
                import traceback
                msg = "exception while computing log-likelihood:\n"
                msg += "   current parameters:\n%s\n" %str(self.fit_params)
                msg += "   traceback:\n%s" %(traceback.format_exc())
                raise RuntimeError(msg)
            
            return lp + lnlike
                        
    def set_fiducial(self):
        """
        Set the fiducial values as the current values of the free parameters
        """
        free = self.theory.free_names
        params = self.theory.fit_params
        theta = np.array([params[key].fiducial for key in free])
        
        if len(theta) != self.theory.ndim:
            logger.error("Problem set fiducial values; not correct number")
            raise ValueError("Number of fiducial values not equal to number of free params")
        self.theory.set_free_parameters(theta)

    def set_fit_results(self, method='median'):
        """
        Set the free parameters from the results objects and update the model
        """
        if self.results is not None:
            if isinstance(self.results, LBFGSResults):
                theta = self.results.min_chi2_values
            else:
                if method == 'median':
                    theta = self.results.values()
                elif method == 'peak':
                    theta = self.results.peak_values()
                elif method == 'max_lnprob':
                    theta = self.results.max_lnprob_values()
                else:
                    raise ValueError("`method` keyword must be one of ['median', 'peak', 'max_lnprob']")
            self.theory.set_free_parameters(theta)
            
    def data_model_pairs(self):
        """
        Return the data - model pairs for each measurement
        
        Returns
        -------
        toret : list
            A list of (k_data, data, err, model) for each measurement 
            in ``data.measurements``
        """
        # this tells you how to slice the flattened results
        flat_slices = self.data.flat_slices
        
        # get the model and data measurements
        toret = []        
        for i, m in enumerate(self.data):          
            model = self.combined_model[flat_slices[i]]
            toret.append((m.k, m.power, m.error, model))
            
        return toret
    
    #---------------------------------------------------------------------------
    # plotting functions
    #---------------------------------------------------------------------------
    def plot_residuals(self):
        """
        Plot the residuals of the measurements with respect to the model, 
        model - measurement
        """
        import plotify as pfy
        
        # get the data - model pairs (k, model, data, err)
        results = self.data_model_pairs()

        # loop over each measurement
        for i, result in enumerate(results):

            # make the residual
            k, data, errs, model = result
            residual = (data - model) / errs
            
            # make the plot
            if self.mode == 'pkmu':
                lab = r"$\mu = %.2f$" %(self.data[i].identifier)
            else:
                lab = r"$\ell = %d$" %(self.data[i].identifier)
            pfy.plot(k, residual, "o", label=lab)

        # make it look nice
        ax = pfy.gca()
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel("wavenumber k", fontsize=16)
        ax.set_ylabel("residuals (data - model)/error", fontsize=16)
        ax.legend(loc=0, ncol=2)
            
    def plot(self):
        """
        Plot the model and data points for the measurements, plotting the 
        P(k, mu) and multipoles on separate figures
        """
        import plotify as pfy
        
        # get the data - model pairs (k, model, data, err)
        results = self.data_model_pairs()
        
        fig = pfy.figure()
        if self.mode == 'pkmu':
            self._plot_pkmu(fig, results, self.data.combined_mu)
        elif self.mode == 'poles':
            self._plot_poles(fig, results, self.data.combined_ell)

    def _plot_pkmu(self, fig, results, mus):
        """
        Plot the model and data points for any P(k,mu) measurements
        """
        import plotify as pfy
        
        ax = fig.gca()
        ax.color_cycle = 'Paired'

        # set up the normalization
        f = self.theory.fit_params['f'].value
        b1 = self.theory.fit_params['b1'].value
        beta = f/b1
        Pnw_kaiser = lambda k, mu: (1. + beta*mu**2)**2 * b1**2 * self.theory.model.normed_power_lin_nw(k)

        offset = -0.1
        flat_slices = self.data.flat_slices
        for i, result in enumerate(results):
            label=r"$\mu = %.2f$" %(self.data[i].identifier)
            
            # unpack the result
            k_data, data, errs, model = result
            mu = mus[flat_slices[i]]
            
            # plot the model
            norm = Pnw_kaiser(k_data, mu)
            pfy.plot(k_data, model/norm + offset*i)
            
            # plot the measurement
            norm = Pnw_kaiser(k_data, mu)
            pfy.errorbar(k_data, data/norm + offset*i, errs/norm, zorder=2, label=label)

        ncol = 1 if self.data.size < 4 else 2
        ax.legend(loc=0, ncol=ncol)
        ax.xlabel.update(r"$k$ (h/Mpc)", fontsize=14)
        ax.ylabel.update(r"$P^{\ gg} / P^\mathrm{EH} (k, \mu)$", fontsize=16)
        
        args = (self.lnprob(), self.Np, self.Nb, self.reduced_chi2())
        ax.title.update(r'$\ln\mathcal{L} = %.2f$, $N_p = %d$, $N_b = %d$, $\chi^2_\mathrm{red} = %.2f$' %args, fontsize=12)
        
    def _plot_poles(self, fig, results, ells):
        """
        Plot the model and data points for any multipole measurements
        """
        import plotify as pfy
        ax = fig.gca()
        ax.color_cycle = 'Paired'

        # set up the normalization
        f = self.theory.fit_params['f'].value
        b1 = self.theory.fit_params['b1'].value
        beta = f/b1
        mono_kaiser = lambda k: (1. + 2./3*beta + 1./5*beta**2) * b1**2 * self.theory.model.normed_power_lin_nw(k)

        for i, result in enumerate(results):
            
            # unpack the result
            k_data, data, errs, model = result
            
            # plot the model
            norm = mono_kaiser(k_data)
            pfy.plot(k_data, model/norm)
            
            # plot the measurement
            label = self.data[i].label
            norm = mono_kaiser(k_data)
            pfy.errorbar(k_data, data/norm, errs/norm, zorder=2, label=label)

        ell_str = ",".join([str(m.identifier) for m in self.data])
        ax.legend(loc=0)
        ax.xlabel.update(r"$k$ (h/Mpc)", fontsize=14)
        ax.ylabel.update(r"$P^{\ gg}_{\ell=%s} / P^\mathrm{EH}_{\ell=0} (k)$" %(ell_str), fontsize=16)
        
        args = (self.lnprob(), self.Np, self.Nb, self.reduced_chi2())
        ax.title.update(r'$\ln\mathcal{L} = %.2f$, $N_p = %d$, $N_b = %d$, $\chi^2_\mathrm{red} = %.2f$' %args, fontsize=12)
        
    
    