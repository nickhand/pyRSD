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
from .results import EmceeResults
import functools
import copy

logger = logging.getLogger('rsdfit.fitting_driver')
logger.addHandler(logging.NullHandler())

class FittingDriver(object):
    """
    A class to handle the data analysis pipeline, merging together a model, 
    theory, and fitting algorithm
    """
    __metaclass__ = rsd_io.PickeableClass
    
    def __init__(self, param_file, extra_param_file=None, pool=None, chains_comm=None, init_model=True):
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
        kwargs['kmin'] = self.data.kmin
        kwargs['kmax'] = self.data.kmax
        self.theory = GalaxyPowerTheory(param_file, **kwargs)
        
        # generic params
        self.params = ParameterSet.from_file(param_file, tags='driver')
        self.pool = pool
        self.chains_comm = chains_comm
        
        # setup the model for data
        if init_model: self._setup_for_data()
        
        # results are None for now
        self.results = None
        
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
        if model_path is not None:
            if not os.path.exists(model_path):
                raise rsd_io.ConfigurationError('provided model file `%s` does not exist' %model_path)
        else:
            model_path = os.path.join(dirname, model_filename)
        if not os.path.exists(params_path):
            raise rsd_io.ConfigurationError('parameter file `%s` must exist to load driver' %params_path)
            
        init_model = (not os.path.exists(model_path)) and init_model
        driver = cls(params_path, init_model=init_model, **kwargs)
        
        # set the model and results
        if os.path.exists(model_path):
            driver.set_model(model_path)
        if results_file is not None:
            if not os.path.exists(results_file):
                if not os.path.join(dirname, results_file):
                    raise rsd_io.ConfigurationError('specified results file `%s` does not exist' %results_file)
                else:
                    results_file = os.path.join(dirname, results_file)
            driver.results = EmceeResults.from_npz(results_file)
    
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
        # if we are initializing from max-like, first call lmfit
        init_from = self.params.get('init_from', None)
        if init_from == 'max-like':
            init_values = self.do_maximum_likelihood()
        elif init_from == 'fiducial':
            free = self.theory.free_names
            params = self.theory.fit_params
            init_values = [params[key].fiducial for key in free]
            if None in init_values:
                raise ValueError("cannot initialize from fiducial values")
            else:
                init_values = np.array(init_values)
            
        # get the solver function
        kwargs = {}
        solver_name = self.params.get('fitter', 'emcee').lower()
        
        # emcee
        if solver_name == 'emcee':
            solver = emcee_fitter.run
            objective = functools.partial(FittingDriver.lnprob, self)
            
            # add some kwargs to pass too
            kwargs['pool'] = self.pool
            kwargs['chains_comm'] = self.chains_comm
            if init_from in ['max-like', 'fiducial']: 
                kwargs['init_values'] = init_values
            elif init_from == 'previous_run':
                kwargs['init_values'] = self.results.copy()
                
        # lmfit
        elif solver_name == 'lmfit':
            solver = lmfit_fitter.run
            objective = functools.partial(FittingDriver.normed_residuals, self)
            
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
        
        # if there was an exception, print out current parameters
        if exception:
            logger.warning("exception occurred: current parameters:\n %s" %str(self.theory.fit_params))
        
        return exception
    
    def finalize_fit(self, exception, results_file):
        """
        Finalize the fit, saving the results file
        """                
        # save the results as a pickle
        logger.info('Saving the results to `%s`' %results_file)
        self.results.to_npz(results_file)
        
        if not exception:
            self.results.summarize_fit()

    def do_maximum_likelihood(self):
        """
        Compute the maximum likelihood values and return them as an array
        """
        # get the solver and objective
        solver = lmfit_fitter.run
        objective = functools.partial(FittingDriver.normed_residuals, self)
        
        # make params
        params = ParameterSet()
        params.add('lmfit_method', value='leastsq')
        
        logger.info("Computing the maximum likelihood values to use as initialization")
        results, exception = solver(params, self.theory, objective)
        logger.info("...done compute maximum likelihood")
        
        values = results.values()
        del results
        return values

    def set_model(self, filename):
        """
        Set the model, as read from a file
        """
        # set it
        logger.info("Setting the theoretical model from file `%s`" %filename)
        self.theory.set_model(filename)
        
        # make the list of model callables
        self._get_model_callables()
        logger.info("...theoretical model successfully read")

    def _setup_for_data(self):
        """
        Setup the model callables for this set of data
        """
        # initialize the model
        logger.info("Initializing theoretical model")
        self.theory.update_model()
        self.theory.model.initialize()
        logger.info("...theoretical model initialized")
        
        # make the list of model callables
        self._get_model_callables()
                
    def _get_model_callables(self):
        """
        Compute the model callables
        """
        # get the mu/ell mappings
        x = []
        for i, meas in enumerate(self.data.measurements):
            if meas.type == 'pkmu':
                x.append(meas.mu)
            elif meas.type == 'pole':
                x.append([meas.ell]*len(meas.k))
            else:
                raise NotImplementedError("only `pkmu` or `pole` supported right now")
        
        self._ks = self.data.combined_k
        if self.mode == 'pkmu':
            self._mus = np.concatenate(x)
            self._model_callable = self.theory.model_callable(self.mode, self._ks, mu=self._mus)
        else:
            self._ells = np.concatenate(x)
            self._model_callable = self.theory.model_callable(self.mode, self._ks, ell=self._ells)
                    
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
        A list of model values for each `PowerMeasurement` in 
        `self.data.measurements`
        """
        N = len(self.data.measurements)
        result = self._model_callable()
        return result.reshape((N, -1)).T
    
    @property
    def combined_model(self):
        """
        Return the model values for each measurement concatenated into a single
        array
        """
        return self.model.ravel(order='F')
        
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
    # Probability functions
    #---------------------------------------------------------------------------
    def lnprior(self):
        """
        Return the log of the prior, based on the current values of the free 
        parameters in  `GalaxyPowerTheory`
        """
        return self.theory.lnprior

    def normed_residuals(self, theta=None):
        """
        Return an array of the normed residuals: (model - data) / diag(covariance)**0.5
        """
        # set the free parameters
        if theta is not None:
            self.theory.set_free_parameters(theta)
        
        norm = np.sqrt(self.data.covariance.diag())
        return  (self.combined_model - self.data.combined_power) / norm

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
        
    def lnlike(self):
        """
        The log of the likelihood, equal to -0.5 * chi2
        """
        return -0.5*self.chi2()
        
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
            good_model = self.theory.set_free_parameters(theta)
            if not good_model:
                return -np.inf
                
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        else:
            toret = lp + self.lnlike() 
            return toret if not np.isnan(toret) else -np.inf

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

    def set_fit_results(self):
        """
        Set the free parameters from the results objects and update the model
        """
        if self.results is not None:
            theta = self.results.values()
            self.theory.set_free_parameters(theta)
            
    def data_model_pairs(self):
        """
        Return the data - model pairs for each measurement
        
        Returns
        -------
        toret : list
            A list of (k_model, model, k_data, data, err) for each measurement in 
            `self.data.measurements`
        """
        N = len(self.data.measurements)
        ks = self._ks.reshape((N, -1)).T
        model = self.model

        # get the model and data measurements
        toret = []
        for i, m in enumerate(self.data.measurements):          
            toret.append((ks[...,i], model[...,i], m.k, m.power, m.error))
            
        return toret
    
    #---------------------------------------------------------------------------
    # Plotting functions
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
            k, model, _, data, errs = result
            residual = (data - model) / errs
            
            # make the plot
            mu = np.mean(self.data.measurements[i].mu)
            pfy.plot(k, residual, "o", label=r"$\mu = %.4f$" %mu)

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
        
        # loop over each measurement
        mus, ells = [], []
        for i, m in enumerate(self.data.measurements):
            if self.mode == 'pkmu':
                mus.append(np.mean(m.mu))
            elif self.mode == 'poles':
                ells.append(m.ell)

        fig = pfy.figure()
        if self.mode == 'pkmu':
            self._plot_pkmu(fig, results, mus)
        elif self.mode == 'poles':
            self._plot_poles(fig, results, ells)

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
        for i in range(len(mus)):
            
            mu = mus[i]
            label=r"$\mu = %.4f$" %mu
            
            # unpack the result
            k_model, model, k_data, data, errs = results[i]
            
            # plot the model
            norm = Pnw_kaiser(k_model, mu)
            pfy.plot(k_model, model/norm + offset*i)
            
            # plot the measurement
            norm = Pnw_kaiser(k_data, mu)
            pfy.errorbar(k_data, data/norm + offset*i, errs/norm, zorder=2, label=label)

        ncol = 1 if len(mus) < 4 else 2
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

        for i, ell in enumerate(ells):
            
            # unpack the result
            k_model, model, k_data, data, errs = results[i]
            
            # plot the model
            norm = mono_kaiser(k_model)
            pfy.plot(k_model, model/norm)
            
            # plot the measurement
            if ell == 0:
                label = "monopole"
            elif ell == 2:
                label = "quadrupole"
            elif ell == 4:
                label = 'hexadecapole'
            else:
                raise ValueError("Do not understand `ell` value for plotting purposes")
            norm = mono_kaiser(k_data)
            pfy.errorbar(k_data, data/norm, errs/norm, zorder=2, label=label)

        ell_str = ",".join(map(str, ells))
        ax.legend(loc=0)
        ax.xlabel.update(r"$k$ (h/Mpc)", fontsize=14)
        ax.ylabel.update(r"$P^{\ gg}_{\ell=%s} / P^\mathrm{EH}_{\ell=%s} (k)$" %(ell_str, ell_str), fontsize=16)
        
        args = (self.lnprob(), self.Np, self.Nb, self.reduced_chi2())
        ax.title.update(r'$\ln\mathcal{L} = %.2f$, $N_p = %d$, $N_b = %d$, $\chi^2_\mathrm{red} = %.2f$' %args, fontsize=12)
        
    
    