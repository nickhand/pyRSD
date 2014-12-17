"""
 fitter.py
 pyRSD/fit: class that fits the RSD model to a measurement using Monte Carlo
 parameter estimation (from `emcee`)
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 12/13/2014
"""
from . import ParamDict
from .. import rsd, numpy as np, os
    
import emcee
import scipy.optimize as opt
import functools
import collections
import pickle
from pandas import DataFrame, Index

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
    def __init__(param_file, num_walkers=20, verbose=False, fig_verbose=False):
        
        # read the parameters
        params = ParamDict(param_file)
        self.tag = params['tag']
        
        # initialize the output
        if not os.path.exists("output_%s" %self.tag):
            os.path.makedirs("output_%s" %self.tag)
                
        # store the info about parameters
        self.pars = ModelParameters(params['parameters'])
        
        # number of total and free parameters
        self.N_total = len(self.pars)
        self.N_free = sum(self.pars(i)['free'] for i in range(self.N_total))
        
        # check for unbounded parameter priors
        for par in self.pars:
            bounds = self.pars[par]['bounds']
            if bounds[0] is None: self.pars[par][0] = -np.inf
            if bounds[1] is None: self.pars[par][0] = np.inf
        
        # load the measurement
        self._load_measurement(params)
            
        # initialize the model
        self._initialize_model(params)

        # find maximum likelihood solution
        self.find_ml_solution()
        
        # initialize the sampler (must be last)
        self.num_walkers = params['walkers']
        self.num_iters = params['iterations']
        self.sampler = emcee.EnsembleSampler(self.num_walkers, self.N_free, functools.partial(GalaxyRSDFitter.lnprob, self))
    
    #end __init__
    
    #---------------------------------------------------------------------------
    def _load_measurement(self, params):
        """
        Load the measurements
        """
        meas = params['measurements']
        self.measurement_names = []
        data = {}
        for i, stat in enumerate(params['statistics']):
            
            # load the data file for this statistic
            this_stat = np.loadtxt(meas[stat])
            
            # make the index or check for consistency
            if i == 0:
                index = Index(data[:, meas[stat]['xcol']], name='k')
            else:
                assert np.all(index == data[:, meas[stat]['xcol']])
                
            # now add the y values
            if stat == 'pkmu':
                for i, mu in enumerate(params['mus']):
                    tag = 'pkmu_%s' %mu
                    self.measurement_names.append(tag)
                    data[tag] = this_stat[:, meas[stat]['ycol'][i]]
                
            else:
                self.measurement_names.append(stat)
                data[stat] = this_stat[:, meas[stat]['ycol']]

        # make a data frame
        self.measurements = DataFrame(d=data, index=index)
        self.measurement_y = np.concatenate([self.measurements[col] for col in self.measurement_names])
        
        # now load the covariance matrix
        self.C = pickle.load(open(params['covariance_matrix'], 'r'))
        ndim = len(self.measurements.columns)*self.measurements.ndim
        if self.C.shape != (ndim, ndim):
            raise ValueError("loaded covariance matrix has wrong shape; expected (%d, %d), got %s" \
                                %(ndim, ndim, self.C.shape))
        
        # rescale by volume
        volume_factor = (params['covariance_vol'] / params['measurement_vol'])**3
        self.C *= volume_factor / params['N_realizations']    
        
        # invert the covariance matrix
        self.C_inv = np.linalg.inv(self.C)
    
    #end _load_measurement
     
    #---------------------------------------------------------------------------
    def _initialize_model(self, params):
        """
        Initialize the model
        """
        # initialize the galaxy power spectrum class
        kwargs = {k:params[k] for k in power_gal.GalaxySpectrum.allowable_kwargs if k in params}
        kwargs['k'] = np.array(self.measurements.index)
        self.model = rsd.power_gal.GalaxySpectrum(**kwargs)
        
        # determine the model callables -- function that returns model at set of k
        self.model_callables = []
        for stat in params['statistics']:
            if stat == 'pkmu':
                self.model_callables.append(functools.partial(getattr(self.model, 'Pgal'), params['mus'], flatten=True))
            elif stat == 'monopole':
                self.model_callables.append(getattr(self.model, 'Pgal_mono'))
            elif stat == 'quadrupole':
                self.model_callables.append(getattr(self.model, 'Pgal_quad'))
            else:
                assert stat in ['monopole', 'quadrupole'], "Statistic must be one of ['pkmu', 'monopole', 'quadrupole']"
                
        # initialize the model
        self.model.initialize()
    #end _initialize_model
    
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
        Return the log of the prior, assuming uniform priors on all parameters
        """
        value_array = self.all_param_values(theta)
        cond = [(self.pars(i)['bounds'][0] < value_array[i] < self.pars(i)['bounds'][1]) for i in range(self.N_total)]  
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
    def find_ml_solution(self, theta):
        """
        Find the parameters that maximizes the likelihood
        """
        # set up the fit
        chi2 = lambda theta: -2 * self.lnlike(theta)
        free_fiducial = np.array([self.pars(i)['fiducial'] for i in range(self.N_total) if self.pars(i)['free']])
        
        # get the max-likelihood values
        result = opt.minimize(chi2, free_fiducial)
        self.ml_values = self.all_param_values(result['x'])
        self.ml_free = result["x"]
        self.ml_minchi2 = self.lnlike(result["x"])*(-2.)
    #end find_ml_solution
    
    #---------------------------------------------------------------------------
    def run(self):
        """
        Run the MCMC sampler
        """
        if params['save_chains']:
            f = open("chain.dat", "w")
            f.close()

            pos0 = [self.free_ml + 1e-4*np.random.randn(self.N_free) for i in range(self.num_walkers)]
            for result in self.sampler.sample(pos0, iterations=self.num_iters, storechain=False):
                position = result[0]
                f = open("chain.dat", "a")
                for k in range(position.shape[0]):
                    f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
            f.close()
        else:
            self.sampler.run_mcmc(pos, self.num_iters, rstate0=np.random.get_state())
    #---------------------------------------------------------------------------