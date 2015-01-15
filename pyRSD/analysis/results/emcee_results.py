from ... import numpy as np
from . import triangle

import plotify as pfy
from matplotlib.ticker import MaxNLocator

#-------------------------------------------------------------------------------
class EmceeParameter(object):
    """
    Class to hold the parameter fitting result
    """
    def __init__(self, trace, burnin=0):
        
        self._trace = trace # shape is (nwalkers, niters)
        self.burnin = burnin
        
    #---------------------------------------------------------------------------
    @property
    def flat_trace(self):
        """
        Returns the flattened chain, excluding steps that occured during the
        "burnin" period
        """
        return self._trace[:, self.burnin:].flatten()
        
    #---------------------------------------------------------------------------
    @property
    def median(self):
        """
        Return the median of the trace, i.e., the 50th percentile
        """
        return np.percentile(self.flat_trace, 50.)
    
    mean = median
    
    #---------------------------------------------------------------------------
    @property
    def one_sigma(self):
        """
        Return the lower and upper one-sigma error intervals, as computed from
        the percentiles, `50 - 15.86555` and `84.13445 - 50`
        
        Returns
        -------
        upper, lower
            The lower and upper 1-sigma error intervals 
        """
        low = 50. - 15.86555
        high = 84.13445 - 50.
        return np.percentile(self.flat_trace, [high, low])

    #---------------------------------------------------------------------------
    @property
    def two_sigma(self):
        """
        Return the lower and upper one-sigma error intervals, as computed from
        the percentiles, `50 - 2.2775` and `84.13445 - 50`

        Returns
        -------
        upper, lower
            The lower and upper 1-sigma error intervals 
        """
        low = 50. - 2.2775
        high = 97.7225 - 50.
        return np.percentile(self.flat_trace, [high, low])
    
    #--------------------------------------------------------------------------- 
    def trace(self, niter=None):
        """
        Return the sample values at a specific iteration number. The output 
        should have a length equal to the number of walkers
        """
        if niter is None:
            return self._trace
        else:
            return self._trace[:,niter]
    
    #---------------------------------------------------------------------------
#endclass EmceeParameter

#-------------------------------------------------------------------------------
class EmceeResults(object):
    """
    Class to hold the fitting results from an `emcee` MCMC run
    """
    def __init__(self, sampler, fit_params, burnin=None):
        """
        Initialize with the `emcee` sampler and the fitting parameters
        """      
        # store the parameter names
        self.free_parameter_names = fit_params.free_parameter_names
        self.constrained_parameter_names = fit_params.constrained_parameter_names
        
        # sampler attributes
        self.chain = sampler.chain # shape is (nwalkers, niters, npars)
        self.lnprobs = sampler.lnprobability # shape is (nwalkers, niters)
        
        # make the constrained chain
        self._make_constrained_chain(fit_params)
        
        # make result params
        self._save_results()
        
        # set the burnin
        if burnin is None: burnin = int(0.1*self.iterations)
        self.burnin = burnin
        
    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        
        # check if key is the name of a free or constrained param
        if key in (self.free_parameter_names + self.constrained_parameter_names):
            return self._results[key]
        else:
            return getattr(self, key)
                    
    #---------------------------------------------------------------------------
    def _make_constrained_chain(self, fit_params):
        """
        Make the chain for the constrained parameters
        """
        if len(self.constrained_parameter_names) == 0:
            self.constrained_chain = None
            return
        
        # make the constrained chain from the other chain
        shape = (self.walkers, self.iterations, len(self.constrained_parameter_names))
        self.constrained_chain = np.zeros(shape)
        for niter in xrange(self.iterations):
            for nwalker, free_param_vals in enumerate(self.chain[:,i,:]):
                
                # set the free parameters
                fit_params.set_free_parameters(free_param_vals)
                
                # set the constrained vals
                constrained_vals = np.array([fit_params[name].value for name in fit_params.constrained_parameter_names])
                self.constrained_chain[nwalker, niter, :] = constrained_vals
            
    #---------------------------------------------------------------------------
    def _save_results(self):
        """
        Make the dictionary of `EmceeParameters`
        """
        self._results = {}
        
        # the free parameters
        for i, name in enumerate(self.free_parameter_names):
            self._results[name] = EmceeParameter(self.chain[...,i])
        
        # the constrained parameters
        if self.constrained_chain is not None:
            for i, name in enumerate(self.constrained_parameter_names):
                self._results[name] = EmceeParameter(self.constrained_chain[...,i])
                        
    #---------------------------------------------------------------------------
    # some convenience attributes
    #---------------------------------------------------------------------------        
    @property
    def iterations(self):
        """
        The number of iterations performed, as computed from the chain
        """
        return self.chain.shape[1]
        
    #---------------------------------------------------------------------------
    @property
    def walkers(self):
        """
        The number of walkers, as computed from the chain
        """
        return self.chain.shape[0]
        
    #---------------------------------------------------------------------------
    @property
    def ndim(self):
        """
        The number of free parameters
        """
        return len(self.free_parameter_names)
        
    #---------------------------------------------------------------------------
    @property
    def burnin(self):
        """
        The number of iterations to treat as part of the "burnin" period, where
        the chain hasn't stabilized yet
        """
        return self._burnin
    
    @burnin.setter
    def burnin(self, val):
        self._burnin = val
        
        for param in self._results:
            self._results[param].burnin = val
    
    #---------------------------------------------------------------------------
    def _plot_timeline(self, names):
        """
        Internal method to make plot timelines for a set of parameters
        """
        pfy.clf()
        fig, axes = pfy.subplots(len(names), 1, sharex=True, figsize=(8, 9))

        # plot timeline for each free parameter
        for i, name in enumerate(names):
            param = self[name]
            
            # this does not exclude a burnin period
            axes[i].plot(param.trace().T, color="k", alpha=0.4)
            axes[i].yaxis.set_major_locator(MaxNLocator(5))
            axes[i].axhline(param.mean, color="#888888", lw=2)
            axes[i].set_ylabel(name)
            
    #---------------------------------------------------------------------------
    def plot_free_timelines(self):
        """
        Plot the timeline chains of the free parameters
        """
        self._plot_timeline(self.free_parameter_names)
        
    #---------------------------------------------------------------------------
    def plot_constrained_timelines(self):
        """
        Plot the timeline chains of the constrained parameters
        """
        if len(self.constrained_parameter_names) > 0:
            self._plot_timeline(self.constrained_parameter_names)
        
    #---------------------------------------------------------------------------
    def plot_free_triangle(self):
        """
        Make the triange plot for the free parameters
        """
        samples = self.chain[:, self.burnin:, :].reshape((-1, self.ndim))
        pfy.clf()
        fig = pfy.gcf()
        fig = triangle.corner(samples, labels=self.free_parameter_names, fig=fig)
        return fig
        
    #---------------------------------------------------------------------------
    def plot_constrained_triangle(self):
        """
        Make the triange plot for the constrained parameters
        """
        N = len(self.constrained_parameter_names)
        samples = self.constrained_chain[:, self.burnin:, :].reshape((-1, N))
        pfy.clf()
        fig = pfy.gcf()
        fig = triangle.corner(samples, labels=self.constrained_parameter_names, fig=fig)
        return fig
        
    #---------------------------------------------------------------------------
#endclass EmceeResults

#-------------------------------------------------------------------------------
    