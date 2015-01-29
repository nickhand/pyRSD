from .. import numpy as np, plotify as pfy, pygcl 
from .parameters import ParameterSet
from .theory import GalaxyPowerTheory
from .data import PowerData
from .fitters import *

import functools
import logging
import cPickle
import copy_reg
import types
import sys

logger = logging.getLogger('pyRSD.analysis.driver')
logger.addHandler(logging.NullHandler())

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
def load_driver(filename):
    """
    Load a driver function
    """
    return cPickle.load(open(filename, 'r'))

#-------------------------------------------------------------------------------
class DataAnalysisDriver(object):
    """
    A class to handle the data analysis pipeline, merging together a model, 
    theory, and fitting algorithm
    """
    def __init__(self, param_file, theory_param_file, data_param_file, pool=None):
        """
        Initialize the driver and the parameters
        """        
        # initialize the data too
        self.data = PowerData(data_param_file)
        
        # initialize the theory
        k_obs = self.data.measurements[0].k
        self.theory = GalaxyPowerTheory(theory_param_file, k=k_obs)
        
        # generic params
        self.params = ParameterSet(param_file)
        self.pool = pool
        
        # setup the model for data
        self._setup_for_data()
        
        # results are None for now
        self.results = None
        
        # pickle instance methods
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
        
    #---------------------------------------------------------------------------
    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_model_callables')
        d['pool'] = None
        return d
        
    def __setstate__(self, d):
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
            ml_values = self.do_maximum_likelihood()
            
        # get the solver function
        kwargs = {}
        solver_name = self.params.get('fitter', 'emcee').lower()
        
        # emcee
        if solver_name == 'emcee':
            solver = emcee_fitter.run
            objective = functools.partial(DataAnalysisDriver.lnprob, self)
            
            # add some kwargs to pass too
            kwargs['pool'] = self.pool
            if init_from == 'max-like': kwargs['ml_values'] = ml_values
        # lmfit
        elif solver_name == 'lmfit':
            solver = lmfit_fitter.run
            objective = functools.partial(DataAnalysisDriver.normed_residuals, self)
        
        # try again
        else:
            raise NotImplementedError("Fitter with name '{}' not currently available".format(solver_name))
          
        # run the solver and store the results
        logger.info("Calling the '{}' fitter's solve function".format(solver_name))
        self.results, exception = solver(self.params, self.theory, objective, **kwargs)
        logger.info("...fitting complete")
        
        try:
            self.set_fit_results()
        except:
            pass
        
        # pickle everything (do this even if an exception was raised)
        output_label = "{}/{}".format(self.params['output_dir'].value, self.params['label'].value)
        self.save(output_label + ".results.pickle")
        
        if exception:
            logger.error("Exception raised...exiting")
            sys.exit()
        else:
            # set the results and summarize
            make_plots = self.params.get('make_plots', False)
            self.results.summarize_fit(output_label, make_plots=make_plots)

    #---------------------------------------------------------------------------
    def do_maximum_likelihood(self):
        """
        Compute the maximum likelihood values and return them as an array
        """
        # get the solver and objective
        solver = lmfit_fitter.run
        objective = functools.partial(DataAnalysisDriver.normed_residuals, self)
        
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
    def save(self, filename):
        """
        Save the class as a pickle
        """
        logger.info("Saving driver class as pickle to filename '{}'".format(filename))
        
        # make sure pool is None, so it is pickable
        self.pool = None
        cPickle.dump(self, open(filename, 'w'))
        
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
        mus, dmus, ells = [], [], []
        for i, meas in enumerate(self.data.measurements):
            if meas.type == 'pkmu':
                mus.append(meas.mu) 
                if meas.dmu is not None: dmus.append(meas.dmu)      
            elif meas.type == 'pole':
                ells.append(meas.ell)
        inds = np.argsort(mus)
        self._mus = [mus[i] for i in inds]
        if len(dmus) == len(mus):
            self._dmus = [dmus[i] for i in inds]
        self._ells = sorted(ells)
        
        # initialize
        self._model_callables = {}
        self._model_callables_hires = {}
        
        # pkmu
        if len(self._mus) > 0:
            kwargs = {}
            if hasattr(self, '_dmus'): kwargs['dmu'] = self._dmus
            self._model_callables['pkmu'] = self.theory.model_callable('pkmu', self._mus, **kwargs)
            self._model_callables_hires['pkmu'] = self.theory.model_callable('pkmu', self._mus, hires=True, **kwargs)
        
        # multipoles
        if len(self._ells) > 0:
            self._model_callables['pole'] = self.theory.model_callable('pole', self._ells)
            self._model_callables_hires['pole'] = self.theory.model_callable('pole', self._ells, hires=True)
                
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
            self.theory.set_free_parameters(theta)
        
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
        theta = np.array([params[key].fiducial_value for key in free])
        
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
            if m.type == 'pkmu':
                index = self._mus.index(m.mu)
                model = pkmu_results[index]
            else:
                index = self._ells.index(m.ell)
                model = pole_results[index]
            
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
        for i, mu in enumerate(mus):
            # unpack the result
            k_model, model, k_data, data, errs = results[i]
            
            # plot the model
            norm = Pnw_kaiser(k_model, mu)
            pfy.plot(k_model, model/norm + offset*i)
            
            # plot the measurement
            norm = Pnw_kaiser(k_data, mu)
            pfy.errorbar(k_data, data/norm + offset*i, errs/norm, zorder=2, label=r"$\mu = %s$" %mu)

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
#endclass DataAnalysisDriver

#-------------------------------------------------------------------------------
        
    
    