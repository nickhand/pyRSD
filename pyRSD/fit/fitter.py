"""
 fitter.py
 pyRSD/fit: class that fits the RSD model to a measurement using Monte Carlo
 parameter estimation (from `emcee`)
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 12/13/2014
"""
from . import param_reader, triangle
from .. import numpy as np, os
from ..rsd import power_gal
 
import copy_reg
import types   
import emcee
import scipy.optimize as opt
import functools
import collections
import pickle
from pandas import DataFrame, Index

from matplotlib.ticker import MaxNLocator
try:
    import plotify as plt
except:
    import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)
 
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
def normalize_covariance_matrix(covar):
    """
    Return the correlation matrix from a covariance matrix

    Parameters
    ----------
    covar : numpy.ndarray, shape (N, N)
        the covariance matrix to normalize
    """
    N, N = covar.shape
    corr = covar.copy()
    
    # normalize the covariance matrix now
    variances = np.diag(corr).copy()
    i, j = np.triu_indices(N)
    corr[i, j] /= np.sqrt(variances[i])*np.sqrt(variances[j])
    
    i, j = np.tril_indices(N, k=-1)
    corr[i, j] /= (np.sqrt(variances[i])*np.sqrt(variances[j]))
    
    return corr
#end normalize_covariance_matrix

#-------------------------------------------------------------------------------
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
    def __init__(self, param_file, pool=None, verbose=True, fig_verbose=True):
        """
        Parameters
        ----------
        param_file : str
            The name of the file holding the parameters to read
        pool : int, optional
            The MPI pool
        verbose : bool, optional
            If `True`, print info about the fit to standard output
        fig_verbose : bool, optional
            If `True`, make a set of figures describing the fit
        """
        # read the parameters
        params = param_reader.ParamDict(param_file)
        
        # store some useful parameters
        self.tag         = params['tag']
        self.save_chains = params['save_chains']
        self.N_walkers   = params['walkers']
        self.N_iters     = params['iterations']
        self.burnin      = params['burnin']
        self.verbose     = verbose
        self.fig_verbose = fig_verbose
        self.pool        = pool    
            
        # initialize the output directory
        if not os.path.exists("output_%s" %self.tag):
            os.makedirs("output_%s" %self.tag)
                
        # initialize the fitting parameters
        self._initialize_fitting_params(params)
        
        # load the measurement
        self._load_measurement(params)
            
        # initialize the model
        if self.verbose: print "Initializing the model..."
        self._initialize_model(params)
        if self.verbose: print "...done"
        
        # pickling methods
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    
    #end __init__
    
    #---------------------------------------------------------------------------
    # utility attributes and functions
    #---------------------------------------------------------------------------
    @property
    def all_param_names(self):
        """
        String names of all the parameter (fixed + free), in the correct order
        """
        return self.pars.keys()
    
    #---------------------------------------------------------------------------
    @property
    def mcmc_free(self):
        """
        MCMC values for free parameters
        """
        return np.array([self.mcmc_fits[par]['mean'] for par in self.free_param_names])
        
    @property
    def mcmc_all_dict(self):
        """
        MCMC values for all parameters in dict form
        """
        return dict((par, self.mcmc_fits[par]['mean']) for par in self.all_param_names)
    
    #---------------------------------------------------------------------------
    def reshape_pkmu(self, pkmu):
        """
        Reshape the P(k,mu) into a data array with shape 
        `(len(self.ks), len(self.mus))`
        """
        return pkmu.reshape(len(self.ks), len(self.mus), order='F')
        
    #---------------------------------------------------------------------------
    def model_kwargs(self, theta):
        """
        Return a dictionary of parameter names and values
        """
        size = len(theta)
        if size == self.N_total:
            return dict(zip(self.all_param_names, theta))
        elif size == self.N_free:
            return dict(zip(self.free_param_names, theta))
        else:
            raise ValueError("Size error in trying to compute model kwargs")
    
    #---------------------------------------------------------------------------
    def all_param_values(self, theta):
        """
        From an array of free parameters `theta`, determine the full parameter
        array values
        """
        i = iter(theta)
        return np.array([i.next() if self.is_free[j] else self.fiducial[j] for j in xrange(self.N_total)])

    #---------------------------------------------------------------------------
    def get_model_fit(self, theta, data_name):
        """
        Return the model fit for the statistic specified by `data_name`, first
        updating the model parameter to those sepcified in `theta`
        
        Returns
        -------
        (k, model) : tuple
            tuple of the k array and model fit at these wavenumbers
        (x, y, yerr) : tuple
            tuple of the x, y, and errors of the data for this statistic
        """
        # update the model
        self.model.update(**self.model_kwargs(theta))
        
        # get the index of this statistics
        if 'pkmu' in data_name:
            i = self.statistics.index('pkmu')
        else:
            i = self.statistics.index(data_name)
        
        # compute the model
        model = self.model_callables[i]()
        if 'pkmu' in data_name:
            imu = self.mus.index(float(data_name.split('_')[-1]))
            model = self.reshape_pkmu(model)[:,imu]
                
        # get the variances
        size = len(self.ks)
        j = self.data_names.index(data_name)
        yerrs = np.diag(self.C**0.5)[size*j : size*(j+1)]
        
        return (self.ks, model), (self.ks, np.array(self.data[data_name]), yerrs)
        
    #---------------------------------------------------------------------------
    def get_model_residuals(self, theta, data_name):
        """
        Return the residuals (data - model) / error
        """
        (_, model), (_, y, yerr) = self.get_model_fit(theta, data_name)
        return (y - model) / yerr
        
    #---------------------------------------------------------------------------
    # Initialization functions
    #---------------------------------------------------------------------------
    def _initialize_fitting_params(self, params):
        """
        Initialize the fitting parameters
        """
        # store the info about parameters
        self.pars = ModelParameters()
        for k in sorted(params['parameters'].keys()):
            self.pars[k] = params['parameters'][k]
        
        # total number of params
        self.N_total = len(self.pars)
        
        # check for unbounded parameter priors
        for par in self.pars:
            bounds = self.pars[par]['bounds']
            if bounds[0] is None: self.pars[par][0] = -np.inf
            if bounds[1] is None: self.pars[par][0] = np.inf
            
        # store some convenience attributes
        self.bounds = np.array([self.pars[par]['bounds'] for par in self.pars])
        self.is_free = np.array([self.pars[par]['free'] for par in self.pars])
        self.fiducial = np.array([self.pars[par]['fiducial'] for par in self.pars])
        
        # free parameter attributes
        self.free_indices = [i for i in range(self.N_total) if self.is_free[i]]
        self.free_param_names = [par for par in self.pars if self.pars[par]['free']]
        self.N_free = sum(self.is_free)
        
        if self.verbose:
            param_names = ", ".join(self.free_param_names)
            print "Fitting %d parameters: %s" %(self.N_free, param_names)
    #end _initialize_fitting_params
    
    #---------------------------------------------------------------------------
    def _load_measurement(self, params):
        """
        Load and store the measurements
        """
        data = params['data']
        self.statistics = params['statistics']
        self.data_names = []
        data_dict = {}
        for i, stat in enumerate(params['statistics']):
            
            # load the data file for this statistic
            this_stat = np.loadtxt(data[stat]['file'])
            
            # make the index or check for consistency
            if i == 0:
                x_full = this_stat[:, data[stat]['xcol']]
                if params.get('k_max', None) is not None:
                    inds = np.where(x_full < params['k_max'])
                index = Index(x_full[inds], name='k')
                self.ks = np.array(index)
            else:
                if not np.all(self.ks == data[:, meas[stat]['xcol']]):
                    msg = "Inconsistent wavenumbers for different measurement statistics"
                    raise ValueError(msg)
                
            # now add the y values
            if stat == 'pkmu':
                self.mus = params['mus']
                for i, mu in enumerate(self.mus):
                    tag = 'pkmu_%s' %mu
                    self.data_names.append(tag)
                    y = this_stat[:, data[stat]['ycol'][i]]
                    if params.get('k_max', None) is not None: y = y[inds]
                    data_dict[tag] = y
            else:
                self.data_names.append(stat)
                y = this_stat[:, data[stat]['ycol']][inds]
                if params.get('k_max', None) is not None: y = y[inds]
                data_dict[stat] = y

        # make a data frame
        self.data = DataFrame(data=data_dict, index=index)
        self.data_y = np.concatenate([self.data[col] for col in self.data_names])
        
        if self.verbose:
            data_names = ", ".join(self.data_names)
            print "Loaded %d measurements: %s" %(len(self.data_names), data_names)
        
        # load the covariance matrix
        self._load_covariance(params, x_full)
    #end _load_measurement
    
    #---------------------------------------------------------------------------
    def _load_covariance(self, params, x_full):
        """
        Load the covariance matrix, possibly trimming by an input `k_max` value
        """
        # now load the covariance matrix
        self.C = pickle.load(open(params['covariance_matrix'], 'r'))
        
        # trim the covariance matrix to the correct k_max
        if params.get('k_max', None) is not None:
            N_stats = len(self.data_names)
            x = np.concatenate([x_full for i in range(N_stats)])
            xx, yy = np.meshgrid(x, x)
            inds = np.where((xx <= params['k_max'])*(yy <= params['k_max']))
            self.C = self.C[inds]
            shape = self.C.shape[0]
            self.C = self.C.reshape(int(shape**0.5), int(shape**0.5))
        
        # check that we have the right dimensions
        ndim = len(self.data_names)*len(self.data)
        if self.C.shape != (ndim, ndim):
            raise ValueError("loaded covariance matrix has wrong shape; expected (%d, %d), got %s" \
                                %(ndim, ndim, self.C.shape))
        
        # rescale by volume
        volume_factor = (params['covariance_volume'] / params['data_volume'])**3
        self.C *= volume_factor / params['N_realizations']    
        
        # invert the covariance matrix
        self.C_inv = np.linalg.inv(self.C)
        
        if self.fig_verbose:
            self.plot_covariance()
    #end _load_covariance
    
    #---------------------------------------------------------------------------
    def _initialize_model(self, params):
        """
        Initialize the model
        """
        # initialize the galaxy power spectrum class
        kwargs = {k:params[k] for k in power_gal.GalaxySpectrum.allowable_kwargs if k in params}
        kwargs['k'] = np.array(self.ks)
        self.model = power_gal.GalaxySpectrum(**kwargs)
        
        # determine the model callables -- function that returns model at set of k
        self.model_callables = []
        for stat in params['statistics']:
            if stat == 'pkmu':
                f = functools.partial(getattr(self.model, 'Pgal'), np.array(params['mus']), flatten=True)
                self.model_callables.append(f)
            elif stat == 'monopole':
                f = getattr(self.model, 'Pgal_mono')
                self.model_callables.append(f)
            elif stat == 'quadrupole':
                f = getattr(self.model, 'Pgal_quad')
                self.model_callables.append(f)
            else:
                if stat not in ['monopole', 'quadrupole']:
                    raise ValueError("Statistic must be one of ['pkmu', 'monopole', 'quadrupole']")
                
        # initialize the model (this is the time-consuming step)
        self.model.update(**self.model_kwargs(self.fiducial))
        self.model.initialize()
    #end _initialize_model
    
    #---------------------------------------------------------------------------
    # Plotting and printing functions
    #---------------------------------------------------------------------------
    def plot_covariance(self):
        """
        Plot the correlation matrix (normalized covariance matrix)
        """
        plt.clf()
        corr = normalize_covariance_matrix(self.C)
        colormesh = plt.pcolormesh(corr, vmin=-1, vmax=1)
        plt.colorbar(colormesh)
        plt.savefig("output_%s/correlation_matrix.png" %self.tag)
    #end plot_covariance
    
    #---------------------------------------------------------------------------
    def print_ml_result(self):
        """
        Print out the maximum likelihood result
        """
        if not hasattr(self, 'ml_free'):
            self.find_ml_solution()
    
        print "Maximum likelihood result:"
        for k, v, i in zip(self.free_param_names, self.ml_free, self.free_indices):
            print "   %-12s = %f (fiducial = %f)" %(k, v, self.fiducial[i])
            
        print "   minimum chi2 = %f" %self.ml_minchi2
    #end print_ml_result
    
    #---------------------------------------------------------------------------
    def print_mcmc_result(self):
        """
        Print out the MCMC results
        """
        if not hasattr(self, 'mcmc_fits'):
            self.compute_quantiles()
            
        print "MCMC result: mean (+/-68%) (+/-95%)"
        for name, fit in self.mcmc_fits.iteritems():
            index = self.all_param_names.index(name)
            if self.is_free[index]:
                x = fit['1-sigma']
                y = fit['2-sigma']
                args = (name, fit['mean'], x[0], x[1], y[0], y[1], self.fiducial[index])
                print "   %-12s = %f (+%f -%f) (+%f -%f) (fiducial = %f)" %args
            
    #end print_ml_result
    
    #---------------------------------------------------------------------------
    def write_mcmc_results(self):
        """
        Write the MCMC results to a file
        """
        if not hasattr(self, 'mcmc_fits'):
            self.compute_quantiles()
            
        out = open("output_%s/mcmc_result.dat" %self.tag, 'w')
        out.write("%f %d %d\n" % (self.ml_minchi2, len(self.data_y), self.N_free))
        for name, fit in self.mcmc_fits.iteritems():
            index = self.all_param_names.index(name)
            x = fit['1-sigma']
            y = fit['2-sigma']
            args = (name, fit['mean'], x[0], x[1], y[0], y[1], self.fiducial[index])
            out.write("%-15s: %f %f %f %f %f %f\n" %args)
            
        out.close()
    #end write_mcmc_results
    
    #---------------------------------------------------------------------------
    def plot_fit(self, theta, extra_tag=""):
        """
        Plot the fit for each statistic using the input parameter value array
        """
        for data_name in self.data_names:
            plt.clf()
            
            # do the plot
            (_, model), (k, data, err) = self.get_model_fit(theta, data_name)
            y = data / model
            err = y*err/data
            plt.errorbar(k, y, err)
            
            # make it look nice
            ax = plt.gca()
            ax.axhline(y=1, c='k', ls='--')
            ax.set_xlabel("wavenumber k", fontsize=16)
            ax.set_ylabel(data_name + " data/model", fontsize=16)
            
            if extra_tag != "":
                plt.savefig("output_%s/model_fit_%s_%s.png" %(self.tag, extra_tag, data_name))
            else:
                plt.savefig("output_%s/model_fit_%s.png" %(self.tag, data_name))
    #end plot_fit
    
    #---------------------------------------------------------------------------
    def plot_residuals(self, theta, extra_tag=""):
        """
        Plot the residuals of the fit for each statistic using the 
        input parameter value array
        """
        plt.clf()
        for data_name in self.data_names:
            # do the plot
            res = self.get_model_residuals(theta, data_name)
            plt.plot(self.ks, res, "o", label=data_name)
            
        # make it look nice
        ax = plt.gca()
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel("wavenumber k", fontsize=16)
        ax.set_ylabel("residuals (data - model)/error", fontsize=16)
        ax.legend(loc=0, ncol=2)
        
        if extra_tag != "":
            plt.savefig("output_%s/model_residuals_%s.png" %(self.tag, extra_tag))
        else:
            plt.savefig("output_%s/model_residuals.png" %(self.tag))
    #end plot_residuals
    
    #---------------------------------------------------------------------------
    def make_timeline_plot(self):
        """
        Make the timeline plot of the chain
        """
        plt.clf()
        fig, axes = plt.subplots(self.N_free, 1, sharex=True, figsize=(8, 9))

        # plot for each free parameter
        for i, name in enumerate(self.free_param_names):
            axes[i].plot(self.sampler.chain[:, :, i].T, color="k", alpha=0.4)
            axes[i].yaxis.set_major_locator(MaxNLocator(5))
            axes[i].axhline(self.ml_free[i], color="#888888", lw=2)
            axes[i].set_ylabel(name)

        fig.tight_layout(h_pad=0.0)
        fig.savefig("output_%s/chain_timeline.png" %self.tag)
    #end make_timeline_plot
    
    #---------------------------------------------------------------------------
    def make_triangle_plot(self):
        """
        Make the triange plot for the parameters
        """
        samples = self.sampler.chain[:, self.burnin:, :].reshape((-1, self.N_free))
        fid_free = self.fiducial[self.free_indices]
        
        plt.clf()
        fig = triangle.corner(samples, labels=self.free_param_names, truths=fid_free)
        fig.savefig("output_%s/triangle.png" %self.tag)
    #end make_triangle_plot
    
    #---------------------------------------------------------------------------
    # Likelihood functions
    #---------------------------------------------------------------------------            
    def lnprior(self, theta):
        """
        Return the log of the prior, assuming uniform priors on all parameters
        """
        value_array = self.all_param_values(theta)
        cond = [self.bounds[i][0] < value_array[i] < self.bounds[i][1] for i in xrange(self.N_total)]
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
        self.model.update(**self.model_kwargs(value_array))
        model_values = np.concatenate([f() for f in self.model_callables])

        print model_values
        # this is chi squared
        diff = model_values - self.data_y
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
        toret = lp + self.lnlike(theta) if np.isfinite(lp) else -np.inf 
        print toret
        return toret
    #end lnprob
    
    #---------------------------------------------------------------------------
    def find_ml_solution(self):
        """
        Find the parameters that maximizes the likelihood
        """
        # set up the fit
        chi2 = lambda theta: -2 * self.lnlike(theta)
        free_fiducial = self.fiducial[self.free_indices]
        
        # get the max-likelihood values
        result = opt.minimize(chi2, free_fiducial, method='Nelder-Mead')
        if result['status'] > 1:
            bounds = self.bounds[self.free_indices]
            result = opt.minimize(chi2, free_fiducial, method='L-BFGS-B', bounds=bounds)
            if result['status'] > 1:
                raise Exception(result['message'])
            
        self.ml_values = self.all_param_values(result['x'])
        self.ml_free = result["x"]
        self.ml_minchi2 = self.lnlike(result["x"])*(-2.)
        
        if self.verbose:
            self.print_ml_result()
        
        if self.fig_verbose:
            self.plot_fit(self.ml_free, extra_tag="maxlike")
            self.plot_residuals(self.ml_free, extra_tag="maxlike")
    #end find_ml_solution
    
    #---------------------------------------------------------------------------
    def run_burnin(self):
        """
        Run the burnin steps
        """
        # find maximum likelihood solution, if we haven't yet
        if not hasattr(self, 'ml_free'):
            self.find_ml_solution()
            
        # initialize the sampler
        objective = functools.partial(GalaxyRSDFitter.lnprob, self)
        self.sampler = emcee.EnsembleSampler(self.N_walkers, self.N_free, objective, pool=self.pool)
        
        # run the steps
        if self.verbose: print "Running %d burn-in steps..." %(self.burnin)
        p0 = [self.ml_free + 1e-3*np.random.randn(self.N_free) for i in range(self.N_walkers)]
        pos, prob, state = self.sampler.run_mcmc(p0, self.burnin)
        if self.verbose: print "...done"
        
        return pos, state
    #end run_burnin
    
    #---------------------------------------------------------------------------
    def run(self):
        """
        Run the MCMC sampler
        """        
        # run the burnin    
        pos0, state = self.run_burnin()
             
        # reset the chain to remove the burn-in samples.
        self.sampler.reset()
                            
        # run the MCMC, either saving the chain or not
        if self.verbose: print "Running %d full MCMC steps..." %(self.N_iters)
        if self.save_chains:
            chain_file = "output_%s/chain.dat" %self.tag
            f = open(chain_file, "w")
            f.close()

            for result in self.sampler.sample(pos0, iterations=self.N_iters, rstate0=state):
                position = result[0]
                f = open(chain_file, "a")
                for k in range(position.shape[0]):
                    f.write("{0:4d} {1:s}\n".format(k, " ".join(str(x) for x in position[k])))
            f.close()
        else:
            self.sampler.run_mcmc(pos0, self.N_iters, rstate0=state)
        if self.verbose: print "...done"      
        
        # close the pool processes
        if self.pool is not None: 
            self.pool.close()
            
        # save the results
        self.write_mcmc_results()
            
        # print out info
        if self.verbose:
            print "Mean acceptance fraction: ", np.mean(self.sampler.acceptance_fraction)
            print "Autocorrelation time: ", self.sampler.get_autocorr_time()
            self.print_mcmc_result()
        
        # make plots
        if self.fig_verbose:
            self.make_timeline_plot()
            self.make_triangle_plot()
    #end run
    
    #---------------------------------------------------------------------------
    def compute_quantiles(self):
        """
        Compute the parameter quantiles
        """
        samples = self.sampler.chain.reshape((-1, self.N_free))
        mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0], v[4]-v[1], v[1]-v[3]), \
                zip(*np.percentile(samples, [15.86555, 50, 84.13445, 2.2775, 97.7225], axis=0)))
            
        self.mcmc_fits = {}
        mcmc_iter = iter(mcmc)
        for i, par in enumerate(self.pars):
            if self.is_free[i]:
                fit = mcmc_iter.next()
                self.mcmc_fits[par] = {'mean':fit[0], '1-sigma':[fit[1], fit[2]], '2-sigma':[fit[3], fit[4]]}
            else:
                self.mcmc_fits[par] = {'mean':self.fiducial[i], '1-sigma':[0., 0.], '2-sigma':[0., 0.]}       

    #end compute_quantiles
    #---------------------------------------------------------------------------
    def save_results(self):
        """
        Save the results as a pickle
        """
        pickle.dump([self.model, self.mcmc_all_dict, self.sampler.chain], open("output_%s/results.pickle" %self.tag, 'w'))
    
    #end save_data
    #---------------------------------------------------------------------------

#endclass GalaxyRSDFitter
#-------------------------------------------------------------------------------
