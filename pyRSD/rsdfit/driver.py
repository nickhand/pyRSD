from .. import numpy as np, os
from . import MPILoggerAdapter, logging
from . import params_filename, model_filename

from .parameters import ParameterSet
from .theory import GalaxyPowerTheory
from .data import PowerData
from .solvers import *
from .util import rsd_io
from .results import EmceeResults, LBFGSResults
from ..rsd import GalaxySpectrum
from ..rsd.window import WindowTransfer
from ..rsd.derivatives import PgalGradient

logger = MPILoggerAdapter(logging.getLogger('rsdfit.fitting_driver'))

NMU = 41

def load_results(filename):
    """
    Load a result from file
    """
    try:
        result = EmceeResults.from_npz(filename)
    except:
        result = LBFGSResults.from_npz(filename)
    return result

class FittingDriver(object, metaclass=rsd_io.PickeableClass):
    """
    A class to handle the data analysis pipeline, merging together a model, 
    theory, and fitting algorithm
    """    
    def __init__(self,
                    param_file, 
                    extra_param_file=None, 
                    init_model=True):
        """
        Initialize the driver with the specified parameters
        
        Parameters
        ----------
        param_file : str
            a string specifying the name of the main parameter file
        extra_param_file : str, optional
            a string specifying the name of a file holding extra extra theory parameters
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
        
        # log the DOF
        args = (self.Nb, self.Np, self.dof)
        logger.info("number of degrees of freedom: %d - %d = %d" %args, on=0)
        
        # generic params
        self.params = ParameterSet.from_file(param_file, tags='driver')
                        
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
            if isinstance(results_file, str) and not os.path.exists(results_file):
                if not os.path.join(dirname, results_file):
                    raise rsd_io.ConfigurationError('specified results file `%s` does not exist' %results_file)
                else:
                    results_file = os.path.join(dirname, results_file)
            driver.results = results_file

    
        return driver
        
    def set_restart(self, restart_file, iterations=None):
        """
        Initialize the `restart` mode by loading the results from a previous
        run
        
        Parameters
        ----------
        restart_file : str
            the name of the file holding the results to restart from
        iterations : int, optional
            if `solver_type` is `mcmc`, this gives the the number of additional 
            iterations to run
        """
        # tell driver we are starting from previous run
        self.params['init_from'].value = 'previous_run'
        
        # set the results
        self.results = restart_file
        total_iterations = iterations + self.results.iterations
        
        # the solver name
        solver_name = self.params.get('solver_type', None).lower()
        if solver_name is None:
            raise ValueError("`solver_type` is `None` -- not sure how to restart")
             
        # set the number of iterations to the total sum we want to do
        if solver_name == 'mcmc':
            if iterations is None:
                raise ValueError("please specify the number of iterations to run when restarting")
            self.params.add('iterations', value=total_iterations)
            self.params.add('walkers', value=self.results.walkers)
        
        elif solver_name == 'nlopt':
            options = self.params.get('lbfgs_options', {})
            options['max_iter'] = total_iterations
            self.params.add('lbfgs_options', value=options)
            
    
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
            
    def run(self, pool=None, chains_comm=None):
        """
        Run the whole fitting analysis, from start to finish
        """            
        init_from = self.params.get('init_from', None)
        init_values = None
        
        # check for deprecated init_values
        if init_from == 'max-like':
            raise ValueError("``init_from = max-like`` has been deprecated; use `nlopt` instead")
        if init_from == 'chain':
            raise ValueError("``init_from = chain`` has been deprecated; use `result` instead")
        
        # init from maximum probability solution
        if init_from == 'nlopt':
            init_values = self.find_peak_probability(pool=pool)
            
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
            
            # log the file name
            logger.info("initializing run from previous result: '%s'" %start_from)
                        
        # restart from previous result
        elif init_from  == 'previous_run':   
            init_values = self.results.copy()
            
        # get the solver function
        solver_name = self.params.get('solver_type', None).lower()
        if solver_name is None:
            raise ValueError("`solver_type` is `None` -- not sure what to do")
        
        kwargs = {'pool':pool}
        if init_values is not None:
            kwargs['init_values'] = init_values
        
        # emcee
        if solver_name == 'mcmc':
            
            # do not use any analytic approximations for bounds and priors
            # since MCMC can handle discontinuity in log-prob
            for name in self.theory.free_names:
                self.theory.fit_params[name].analytic = False
                
            solver = emcee_solver.run
            kwargs['chains_comm'] = chains_comm
            
        # lbfgs
        elif solver_name == 'nlopt':
            solver = lbfgs_solver.run
                        
            # use analytic approximation for bounds and priors
            # to enforce continuity
            for name in self.theory.free_names:
                self.theory.fit_params[name].analytic = True
                
        # incorrect
        else:
            raise NotImplementedError("solver with name '{}' not currently available".format(solver_name))
          
        # run the solver and store the results
        if init_from == 'previous_run':
            logger.info("Restarting '{}' solver from a previous result".format(solver_name))
        else:
            logger.info("Calling the '{}' solve function".format(solver_name))
        self.results, exception = solver(self.params, self.theory, **kwargs)
        
        # store the model version in the results
        if self.results is not None:
            self.results.model_version = self.theory.model.__version__
        
        logger.info("...fitting complete")

        return exception
        
    def finalize_fit(self, exception, results_file):
        """
        Finalize the fit, saving the results file
        """                
        # save the results as a pickle
        logger.info('Saving the results to `%s`' %results_file)
        self.results.to_npz(results_file)
        
        # summarize if no exception
        if exception is None or isinstance(exception, KeyboardInterrupt):
            self.results.summarize_fit()
        
    def find_peak_probability(self, pool=None):
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
        solver = bfgs_solver.run
                
        # init values from fiducial
        init_values = self.theory.free_fiducial
        
        logger.info("using L-BFGS soler to find the maximum probability values to use as initialization")
        results, exception = solver(params, self.theory, pool=pool, init_values=init_values)
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
        logger.info("Initializing theoretical model", on=0)
        self.theory.update_model()
        self.theory.model.initialize()
        logger.info("...theoretical model initialized", on=0)
        
        # the model callable
        self._model_callable = self.theory.model_callable(self.data)
                                                
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
        if isinstance(val, str):
            val = load_results(val)
            
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
        if isinstance(val, str):
            logger.info("setting the theoretical model from file `%s`" %val, on=0)
        else:
            logger.info("setting the theoretical model from existing instance", on=0)
        self.theory.set_model(val)
        
        # print out the model paramete
        params = self.theory.model.to_dict()
        msg = "running with model parameters:\n\n"
        msg += "\n".join(["%-25s: %s" %(k, str(v)) for k,v in sorted(params.items())])
        logger.info(msg, on=0)
        
        # model callable
        self._model_callable = self.theory.model_callable(self.data)
        logger.info("...theoretical model successfully read", on=0)
        
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
        return self.data.ndim
        
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
        else:
            in_bounds = all(p.within_bounds() for p in self.theory.free)
            
        # return -np.inf if any parameters are out of bounds
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
                msg += "   current parameters:\n%s\n" %str(self.theory.fit_params)
                msg += "   traceback:\n%s" %(traceback.format_exc())
                raise RuntimeError(msg)
            
            return lp + lnlike
                                
    def minus_lnlike(self, theta=None, use_priors=False):
        """
        Return the negative log-likelihood, optionally including priors
        
        Parameters
        ----------
        theta : array_like, optional
            if provided, the values of the free parameters to compute the probability
            at; it not provided, use the current values of free parameters from 
            `theory.fit_params`
        use_priors : bool, optional
            whether to include the log priors in the objective function when 
            minimizing the negative log probability
        """
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)
            
            # if parameters are out of bounds, return the log-likelihood
            # for a null model + the prior results (which hopefully are restrictive)
            if not in_bounds:
                
                with self.theory.preserve(theta):
                    return -self.null_lnlike - self.lnprior()
        else:
            theta = self.theory.free_values
        
        # value at theta
        prob = -self.lnlike()
        if use_priors:
            prob += -self.lnprior()
            
        return prob
            
    #---------------------------------------------------------------------------
    # gradient calculations
    #---------------------------------------------------------------------------
    @property
    def pkmu_gradient(self):
        """
        Return the P(k,mu) gradient class
        """
        try:
            return self._pkmu_gradient
        except AttributeError:
            
            # evaluate gradient on the transfer grid
            if self.data.transfer is not None:
                
                transfer = self.data.transfer
                grid = transfer.grid
                k, mu = grid.k[grid.notnull], grid.mu[grid.notnull]
            
            # evaluate over own k,mu grid
            else:
                k = self.data.combined_k
                if self.mode == 'poles':
                    mu = np.linspace(0., 1., NMU)
            
                    # broadcast to the right shape
                    k     = k[:, np.newaxis]
                    mu    = mu[np.newaxis, :]
                    k, mu = np.broadcast_arrays(k, mu)
                    k     = k.ravel(order='F')
                    mu    = mu.ravel(order='F')
            
                else:
                    mu = self.data.combined_mu
                
            # make the gradient object       
            self._pkmu_gradient = PgalGradient(self.model, self.theory.fit_params, k, mu)
            return self._pkmu_gradient
                    
    def grad_minus_lnlike(self, theta=None, epsilon=1e-4, pool=None, use_priors=True, 
                            numerical=False, numerical_from_lnlike=False):
        """
        Return the vector of the gradient of the negative log likelihood,
        with respect to the free parameters, optionally evaluating at `theta`
        
        This uses a central-difference finite-difference approximation to 
        compute the numerical derivatives
        
        Parameters
        ----------
        theta : array_like, optional
            if provided, the values of the free parameters to compute the 
            gradient at; if not provided, the current values of free parameters 
            from `theory.fit_params` will be used
        epsilon : float or array_like, optional
            the step-size to use in the finite-difference derivative calculation; 
            default is `1e-4` -- can be different for each parameter
        pool : MPIPool, optional
            a MPI Pool object to distribute the calculations of derivatives to 
            multiple processes in parallel
        use_priors : bool, optional
            whether to include the log priors in the objective function when 
            minimizing the negative log probability
        numerical : bool, optional
            if `True`, evaluate gradients of P(k,mu) numerically using finite difference
        numerical_from_lnlike : bool, optional
            if `True`, evaluate the gradient by taking the numerical derivative
            of :func:`minus_lnlike`
                            
        """                
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)
            
            # if parameters are out of bounds, return the log-likelihood
            # for a null model + the prior results (which hopefully are restrictive)
            if not in_bounds:
                
                # set parameters to get derivatives at invalid state
                with self.theory.preserve(theta):
                    
                    # derivatives at invalid state
                    dlnprior = self.theory.dlnprior
                    
                return -1.0 * dlnprior
        else:
            theta = self.theory.free_values
        
        # numerical gradient of minus_lnlike
        if numerical_from_lnlike:

            increments = np.identity(len(theta)) * epsilon
            tasks = np.concatenate([theta+increments, theta-increments], axis=0)
            results = np.array([self.minus_lnlike(t) for t in tasks])
            results = results.reshape((2, -1))

            grad_minus_lnlike = (results[0] - results[1]) / (2.*epsilon)
        
        else:
        
            # get the gradient of Pgal wrt the parameters
            gradient = self.pkmu_gradient(theta, pool=pool, epsilon=epsilon, numerical=numerical)
        
            # evaluate gradient via the transfer function
            if self.data.transfer is not None:
                
                grad_lnlike = []
                for i in range(self.Np):
                    self.data.transfer.power = gradient[i]
                    grad_lnlike.append(self.data.transfer(flatten=True))    
                grad_lnlike = np.asarray(grad_lnlike)
            
            # do pole calculation yourself
            elif self.mode == 'poles':
                from scipy.special import legendre
                from scipy.integrate import simps

                # reshape the arrays
                k        = self.pkmu_gradient.k.reshape(-1, NMU, order='F')
                mu       = self.pkmu_gradient.mu.reshape(-1, NMU, order='F')
                gradient = gradient.reshape(self.Np, -1, NMU, order='F')

                # do the integration over mu
                kern     = np.asarray([(2*ell+1.)*legendre(ell)(mu[i]) for i, ell in enumerate(self.data.combined_ell)])
                grad_lnlike = np.array([simps(t, x=mu[i]) for i,t in enumerate(kern*gradient)])
                
            else:
                grad_lnlike = gradient

            # transform from model gradient to log likelihood gradient
            diff = self.data.combined_power - self.combined_model
            grad_lnlike = np.dot(np.dot(self.data.covariance.inverse, diff), grad_lnlike.T)
            grad_minus_lnlike = -1 * grad_lnlike

        # test for inf
        has_inf = np.isinf(grad_minus_lnlike)
        if has_inf.any():
            bad = np.array(self.theory.free_names)[has_inf].tolist()
            raise ValueError("inf values detected in the gradient for: %s" %str(bad))

        # test for NaN
        has_nan = np.isnan(grad_minus_lnlike)
        if has_nan.any():
            bad = np.array(self.theory.free_names)[has_nan].tolist()
            raise ValueError("NaN values detected in the gradient for: %s" %str(bad))
            
        # add in log prior prob and derivatives of log priors
        if use_priors: 
            grad_minus_lnlike += -1 * self.theory.dlnprior
            
        return grad_minus_lnlike
        
    #---------------------------------------------------------------------------
    # setting results
    #---------------------------------------------------------------------------
    def set_fiducial(self):
        """
        Set the fiducial values as the current values of the free parameters
        """
        free = self.theory.free_names
        params = self.theory.fit_params
        theta = np.array([params[key].fiducial for key in free])
        
        if len(theta) != self.theory.ndim:
            logger.error("problem setting fiducial values; not correct number")
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
            
    def preserve(self):
        """
        Context manager that preserves the state of the model
        upon exiting the context by first saving and then restoring it
        """
        return self.theory.model.preserve()
            
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
        from matplotlib import pyplot as plt
        
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
            plt.plot(k, residual, "o", label=lab)

        # make it look nice
        ax = plt.gca()
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel("wavenumber k", fontsize=16)
        ax.set_ylabel("residuals (data - model)/error", fontsize=16)
        ax.legend(loc=0, ncol=2)
            
    def plot(self, usetex=False, ax=None, colors=None, use_labels=True, **kws):
        """
        Plot the model and data points for the measurements, plotting the 
        P(k, mu) and multipoles on separate figures
        """
        from .util import plot
        
        # plot the fit comparison
        ax = plot.plot_fit_comparison(self, ax=ax, colors=colors, use_labels=use_labels, **kws)
        
        # set usetex
        if usetex:
            ax.figure.usetex = True
