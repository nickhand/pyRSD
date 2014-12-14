"""
 fitter.py
 pyRSD/fit: class that fits the RSD model to a meausurement using Monte Carlo
 parameter estimation (from `emcee`)
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 12/13/2014
"""
from . import ParamDict
from .. import rsd, numpy as np

import emcee
import scipy.optimize as opt
import functools
import collections

class ModelParameters(collections.OrderedDict):
    """
    A subclass of collections.OrderedDict that adds __getitem__
    """
    def __init__(self, *args, **kwargs):
        super(ModelParameters, self).__init__(*args, **kwargs)
        
    def __call__(self, key):
        if not isinstance(key, int): raise KeyError("key must be an integer")
        if key >= len(self): raise KeyError("key out of range")
        return self[self.keys()[key]]

#-------------------------------------------------------------------------------
class GalaxyRSDFitter(object):
    """
    Subclass of `emcee.EnsembleSampler` to compute parameter fits for RSD models
    """
    def __init__(param_file, num_walkers=20):
        
        # read the parameters
        params = ParamDict(param_file)
                
        # store the info about parameters
        self.pars = ModelParameters(params['parameters'])
        
        # number of total and free parameters
        self.N_total = len(self.pars)
        self.N_free = sum(self.pars(i)['free'] for i in range(self.N_total))
        
        # set prior bounds
        for par in self.pars:
            bounds = self.pars[par]['bounds']
            if bounds[0] is None: self.pars[par][0] = -np.inf
            if bounds[1] is None: self.pars[par][0] = np.inf
        
        # load the measurement
        self._load_measurement()
        
        # load the covariance matrix
        self._load_covariance_matrix()
        
        # initialize the galaxy power spectrum class
        kwargs = {k:params.pop(k) for k in power_gal.GalaxySpectrum.allowable_kwargs if k in params}
        self.model = rsd.power_gal.GalaxySpectrum(**kwargs)
        
        # determine the model callables -- function that returns model at set of k
        self.model_callables = []
        for stat in params['statistics']:
        
            # if it's pkmu, then bind the value of mu to the function call...genius
            if stat not in ['monopole', 'quadrupole']:
                self.model_callables.append(functools.partial(getattr(self.model, 'power'), float(stat)))
            else:
                self.model_callables.append(getattr(self.model, stat))
                
                
        # find maximum likelihood solution
        self.find_ml_solution()
        
        # initialize the sampler (must be last)
        self.num_walkers = num_walkers
        pos = [free_ml + 1.e-4*np.random.randn(self.N_free) for i in range(self.num_walkers)]
        self.sampler = emcee.EnsembleSampler(self.num_walkers, self.N_free, functools.partial(GalaxyRSDFitter.log_prob, self))
    
    #end __init__
    
    #---------------------------------------------------------------------------
    def all_param_values(self, theta):
        """
        From an array of free parameters `theta`, determine the full parameter
        array values
        """
        theta_iter = iter(theta)
        return np.array([theta_iter.next() if self.pars(i)['free'] else self.pars(i)['fiducial'] for i in range(self.N_total)])

    #end full_param_values
    
    #---------------------------------------------------------------------------
    def lnprior(self, theta):
        """
        Return the log of the prior, assuming uniform prior on any parameters
        """
        value_array = self.all_param_values(theta)
        cond = [(self.pars(i)['bounds'][0] < value < self.pars(i)['bounds'][1]) for i, value in enumerate(value_array)]  
        return 0. if all(cond) else -np.inf    
    #end lnprior
    
    #---------------------------------------------------------------------------
    def lnlike(self, theta):
        """
        The log of likelihood function for the specified model function
        """
        # get the array of fixed+free parameters
        value_array = self.all_param_values(theta)
        
        # update the model 
        self.model.update(**{k : value_array[i] for i, k in enumerate(self.pars)})
        model_values = np.concatenate([f() for f in self.model_callables])

        # this is chi squared
        diff = model_values - self.measurement_y
        chi2 = np.dot(diff, np.dot(self.C_inv, diff))
        
        return -0.5*chi2
    #end lnlike
    
    #---------------------------------------------------------------------------
    def lnprob(self, theta):
        """
        The log of the posterior probability function, defined as likelihood * prior
        """
        lp = self.lnprior(theta)
        value_array = self.all_param_values(theta)
        return lp + self.lnlike(theta) if np.isfinite(lp) else -np.inf 
    #end lnprob
    
    #---------------------------------------------------------------------------
        
        