try:
    import plotify as pfy
except:
    raise ImportError("`Plotify` plotting package required for `rsdfit`")
    
from .. import numpy as np, pygcl 
from .parameters import ParameterSet
from .theory import GalaxyPowerTheory
from .data import PowerData
from .fitters import *
from .util import rsd_io

import functools
import logging
import copy_reg
import types

logger = logging.getLogger('rsdfit.fitting_driver')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
def load_driver(params_file, model_file, results_file):
    """
    Load a driver object from a parameter file, results file, and model file
    """
    driver = FittingDriver(params_file, initialize_model=False)
    driver.set_model(model_file)
    driver.results = rsd_io.load_pickle(results_file)
    
    return driver
    
#-------------------------------------------------------------------------------
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)
 
#-------------------------------------------------------------------------------
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)
    
#-------------------------------------------------------------------------------
class FittingDriver(object):
    """
    A class to handle the data analysis pipeline, merging together a model, 
    theory, and fitting algorithm
    """
    def __init__(self, param_file, extra_param_file=None, pool=None, initialize_model=True):
        """
        Initialize the driver and the parameters
        """        
        # initialize the data too
        self.data = PowerData(param_file)
        
        # initialize the theory
        k_obs = self.data.measurements[0].k
        self.theory = GalaxyPowerTheory(param_file, extra_param_file=extra_param_file, k=k_obs)
        
        # generic params
        self.params = ParameterSet(param_file, tag='driver')
        self.pool = pool
        
        # setup the model for data
        if initialize_model:
            self._setup_for_data()
        
        # results are None for now
        self.results = None
        
        # pickle instance methods
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
        
    #---------------------------------------------------------------------------
    def to_file(self, filename, mode='w'):
        """
        Save the parameters of this driver to a file
        """
        # first save the driver params
        self.params.to_file(filename, mode='w', header_name='driver params', 
                            footer=True, as_dict=False)
                            
        # now save the data params
        self.data.to_file(filename, mode='a')
        
        # and now the theory params
        self.theory.to_file(filename, mode='a')
    
    #---------------------------------------------------------------------------
    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_model_callables')
        d['pool'] = None
        return d
        
    def __setstate__(self, d):
        
        # pickle instance methods
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
        
        self.__dict__ = d
        self._get_model_callables()
        
    #---------------------------------------------------------------------------
    def run(self):
        """
        The main function, this will run the whole analysis from start to 
        finish (hopefully)
        """
        # if we are initializing from max-like, first call lmfit
        init_from = self.params.get('init_from', None)
        if init_from == 'max-like':
            init_values = self.do_maximum_likelihood()
        elif init_from == 'fiducial':
            free = self.theory.free_parameter_names
            params = self.theory.fit_params
            init_values = [params[key].fiducial for key in free]
            if None in init_values:
                raise ValueError("Cannot initialize from fiducial values")
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
            if init_from in ['max-like', 'fiducial']: 
                kwargs['init_values'] = init_values
        # lmfit
        elif solver_name == 'lmfit':
            solver = lmfit_fitter.run
            objective = functools.partial(FittingDriver.normed_residuals, self)
        
        # try again
        else:
            raise NotImplementedError("Fitter with name '{}' not currently available".format(solver_name))
          
        # run the solver and store the results
        logger.info("Calling the '{}' fitter's solve function".format(solver_name))
        self.results, exception = solver(self.params, self.theory, objective, **kwargs)
        logger.info("...fitting complete")
        
        return exception

    #---------------------------------------------------------------------------
    def restart_chain(self, iterations, old_chain_file, pool=None):
        """
        Load the old chain given by `old_chain_file`, store as `self.results`, 
        and continue sampling the chain for `iterations` more steps
        
        Parameters
        ----------
        iterations : int 
            the number of iterations to run
        old_chain_file : str
            the old chain file to start from
        pool : emcee.MPIPool, NoneType
            an optional pool of processors to use
        """
        # tell driver we are starting from previous run
        self.params['init_from'].value = 'previous_run'
        
        # load the old chain
        old_chain = rsd_io.load_pickle(old_chain_file)
        
        # set the number of iterations to the total sum we want to do
        self.params.add('iterations', iterations + old_chain.iterations)
        self.params.add('walkers', old_chain.walkers)
        
        # make sure we want to use emcee
        solver_name = self.params.get('fitter', 'emcee').lower()
        if solver_name != 'emcee':
            raise ValueError("Cannot restart chain if desired solver is not `emcee`")
            
        solver = emcee_fitter.run
        objective = functools.partial(FittingDriver.lnprob, self)
        
        # run the solver 
        logger.info("Restarting `emcee` solver from an old chain")
        kwargs = {'pool' : pool, 'init_values' : old_chain}
        results, exception = solver(self.params, self.theory, objective, **kwargs)
        
        # set the results
        self.results = results
        logger.info("...fitting complete")

        return exception
    
    #---------------------------------------------------------------------------
    def finalize_fit(self, exception, results_file):
        """
        Finalize the fit, saving the results file
        """                
        # save the results as a pickle
        logger.info('Saving the results to `%s`' %results_file)
        rsd_io.save_pickle(self.results, results_file)
        
        if not exception:
            self.results.summarize_fit()
            
    #---------------------------------------------------------------------------
    def do_maximum_likelihood(self):
        """
        Compute the maximum likelihood values and return them as an array
        """
        # get the solver and objective
        solver = lmfit_fitter.run
        objective = functools.partial(FittingDriver.normed_residuals, self)
        
        # make params
        params = ParameterSet()
        params['lmfit_method'] = 'leastsq'
        
        logger.info("Computing the maximum likelihood values to use as initialization")
        results, exception = solver(params, self.theory, objective)
        logger.info("...done compute maximum likelihood")
        
        values = results.values()
        del results
        return values

    #---------------------------------------------------------------------------
    def set_model(self, filename):
        """
        Set the model, as read from a file
        """
        # read in the model
        model = rsd_io.load_pickle(filename)
        
        # set it
        logger.info("Setting the theoretical model from file `%s`" %filename)
        self.theory.model = model
        self.theory.update_constraints()
        

        # make the list of model callables
        self._get_model_callables()
        logger.info("...theoretical model successfully read")


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
        
        # make the list of model callables
        self._get_model_callables()
                
    #---------------------------------------------------------------------------
    def _get_model_callables(self):
        """
        Compute the model callables
        """
        # get the mu/ell mappings
        mus, ells = [], []
        for i, meas in enumerate(self.data.measurements):
            if meas.type == 'pkmu':
                mus.append(meas.mu)   
            elif meas.type == 'pole':
                ells.append(meas.ell)
        inds = np.argsort(mus)
        
        self._mus = [mus[i] for i in inds]
        self._ells = sorted(ells)
        
        # initialize
        self._model_callables = {}
        self._model_callables_hires = {}
        
        # pkmu
        if len(self._mus) > 0:
            kwargs = {}            
            self._model_callables['pkmu'] = self.theory.model_callable('pkmu', self._mus, **kwargs)
            self._model_callables_hires['pkmu'] = self.theory.model_callable('pkmu', self._mus, hires=True, **kwargs)

        # multipoles
        if len(self._ells) > 0:
            self._model_callables['pole'] = self.theory.model_callable('pole', self._ells)
            self._model_callables_hires['pole'] = self.theory.model_callable('pole', self._ells, hires=True)
                
    #---------------------------------------------------------------------------
    def save(self, filename):
        """
        Save as a pickle
        """
        logger.info("Saving driver class as pickle to filename '{}'".format(filename))
        
        self.pool = None
        rsd_io.save_pickle(self)
        
    #---------------------------------------------------------------------------
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
            free_params = self.theory.free_parameter_names
            constrained_params = self.theory.constrained_parameter_names
            val.verify_param_ordering(free_params, constrained_params)
            
        self._results = val
        
    #---------------------------------------------------------------------------
    @property
    def model(self):
        """
        A list of model values for each `PowerMeasurement` in 
        `self.data.measurements`
        """
        # split the pkmu/pole results
        if 'pkmu' in self._model_callables:
            pkmu_results = np.split(self._model_callables['pkmu'](), len(self._mus))
        if 'pole' in self._model_callables:
            pole_results = np.split(self._model_callables['pole'](), len(self._ells))
            
        toret = []
        for m in self.data.measurements:
            if m.type == 'pole':
                index = self._ells.index(m.ell)
                toret.append(pole_results[index])
            else:
                index = self._mus.index(m.mu)
                toret.append(pkmu_results[index])
        return toret
        
    #---------------------------------------------------------------------------        
    @property
    def combined_model(self):
        """
        Return the model values for each measurement concatenated into a single
        array
        """
        return np.concatenate(self.model)
        
    #---------------------------------------------------------------------------
    @property
    def dof(self):
        """
        Return the degrees of freedom, equal to number of data points minus
        the number of free parameters
        """
        return len(self.combined_model) - len(self.theory.free_parameters)
        
    #---------------------------------------------------------------------------
    # Probability functions
    #---------------------------------------------------------------------------
    def lnprior(self):
        """
        Return the log of the prior, based on the current values of the free 
        parameters in  `GalaxyPowerTheory`
        """
        return self.theory.lnprior

    #---------------------------------------------------------------------------
    def normed_residuals(self, theta=None):
        """
        Return an array of the normed residuals: (model - data) / diag(covariance)**0.5
        """
        # set the free parameters
        if theta is not None:
            self.theory.set_free_parameters(theta)
        
        norm = np.sqrt(self.data.covariance.diag())
        return  (self.combined_model - self.data.combined_power) / norm
        
    #---------------------------------------------------------------------------
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
    
    #---------------------------------------------------------------------------
    def reduced_chi2(self):
        """
        The reduced chi squared value
        """
        return self.chi2() / self.dof
        
    #---------------------------------------------------------------------------
    def lnlike(self):
        """
        The log of the likelihood, equal to -0.5 * chi2
        """
        return -0.5*self.chi2()
        
    #---------------------------------------------------------------------------
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
            return lp + self.lnlike()
    #end lnprob
    
    #---------------------------------------------------------------------------
    def set_fiducial(self):
        """
        Set the fiducial values as the current values of the free parameters
        """
        free = self.theory.free_parameter_names
        params = self.theory.fit_params
        theta = np.array([params[key].fiducial for key in free])
        
        if len(theta) != self.theory.ndim:
            logger.error("Problem set fiducial values; not correct number")
            raise ValueError("Number of fiducial values not equal to number of free params")
        self.theory.set_free_parameters(theta)
    
    #---------------------------------------------------------------------------
    def set_fit_results(self):
        """
        Set the free parameters from the results objects and update the model
        """
        if self.results is not None:
            theta = self.results.values()
            self.theory.set_free_parameters(theta)
            
    #---------------------------------------------------------------------------
    def data_model_pairs(self, hires=False):
        """
        Return the data - model pairs for each measurement
        
        Returns
        -------
        toret : list
            A list of (k_model, model, k_data, data, err) for each measurement in 
            `self.data.measurements`
        """
        if not hires:
            callables = self._model_callables
        else:
            callables = self._model_callables_hires
            
        # get the pkmu/pole values
        if 'pkmu' in callables:
            pkmu_results = np.split(callables['pkmu'](), len(self._mus))
        if 'pole' in callables:
            pole_results = np.split(callables['pole'](), len(self._ells))

        # get the model and data measurements
        toret = []
        for m in self.data.measurements:
            if m.type == 'pole':
                index = self._ells.index(m.ell)
                model = pole_results[index]
            else:
                index = self._mus.index(m.mu)
                model = pkmu_results[index]
            
            toret.append((self.theory.model.k_obs, model, m.k, m.power, m.error))
            
        return toret
    
    #---------------------------------------------------------------------------
    # Plotting functions
    #---------------------------------------------------------------------------
    def plot_residuals(self):
        """
        Plot the residuals of the measurements with respect to the model, 
        model - measurement
        """
        # get the data - model pairs (k, model, data, err)
        results = self.data_model_pairs()

        pkmu_results, pole_results = [], []
        mus, ells = [], []

        # loop over each measurement
        for i, result in enumerate(results):

            # make the residual
            k, model, _, data, errs = result
            residual = (data - model) / errs
            
            # make the plot
            pfy.plot(k, residual, "o", label=self.data.measurements[i].label)

        # make it look nice
        ax = pfy.gca()
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel("wavenumber k", fontsize=16)
        ax.set_ylabel("residuals (data - model)/error", fontsize=16)
        ax.legend(loc=0, ncol=2)
            
        
    #---------------------------------------------------------------------------
    def plot(self):
        """
        Plot the model and data points for the measurements, plotting the 
        P(k, mu) and multipoles on separate figures
        """
        # get the data - model pairs (k, model, data, err)
        results = self.data_model_pairs(hires=True)
        
        pkmu_results, pole_results = [], []
        mus, ells = [], []
        
        # loop over each measurement
        for i, result in enumerate(results):
            
            m = self.data.measurements[i]
            kind = m.type
            if kind == 'pkmu':
                mus.append(m.mu)
                pkmu_results.append(result)
            else:
                ells.append(m.ell)
                pole_results.append(result)
            
        # plot pkmu
        if len(pkmu_results) > 0:
            fig = pfy.figure()
            self._plot_pkmu(fig, pkmu_results, mus)
            
        # plot poles
        if len(pole_results) > 0:
            fig = pfy.figure()
            self._plot_poles(fig, pole_results, ells)
            
    #---------------------------------------------------------------------------
    def _plot_pkmu(self, fig, results, mus):
        """
        Plot the model and data points for any P(k,mu) measurements
        """
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
            label=r"$\mu = %s$" %mu
            
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
    
    #---------------------------------------------------------------------------
    def _plot_poles(self, fig, results, ells):
        """
        Plot the model and data points for any multipole measurements
        """
        ax = fig.gca()
        ax.color_cycle = 'Paired'

        # set up the normalization
        f = self.theory.model.f; b1 = self.theory.model.b1
        beta = f/b1
        mono_kaiser = lambda k: (1. + 2./3*beta + 1./5*beta**2) * b1**2 * self.theory.model.normed_power_lin(k)

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
            else:
                raise ValueError("Do not understand `ell` value for plotting purposes")
            norm = mono_kaiser(k_data)
            pfy.errorbar(k_data, data/norm, errs/norm, zorder=2, label=label)

        ax.legend(loc=0)
        ax.xlabel.update(r"$k$ (h/Mpc)", fontsize=14)
        ax.ylabel.update(r"$P^{\ gg}_{\ell=0,2} / P^\mathrm{NW}_{\ell=0,2} (k)$", fontsize=16)

    #---------------------------------------------------------------------------
#endclass FittingDriver

#-------------------------------------------------------------------------------
        
    
    