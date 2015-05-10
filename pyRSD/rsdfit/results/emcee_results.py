try:
    import plotify as pfy
except:
    raise ImportError("`Plotify` plotting package required for `rsdfit`")
    
from ... import numpy as np
from . import triangle, tools

from matplotlib.ticker import MaxNLocator
import copy
import logging
import cPickle
import scipy.stats

logger = logging.getLogger('rsdfit.emcee_results')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
class EmceeParameter(object):
    """
    Class to hold the parameter fitting result
    """
    def __init__(self, name, trace, burnin=0):
        
        self.name   = name
        self._trace = trace # shape is (nwalkers, niters)
        self.burnin = int(burnin)
        
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
        sig1 = self.one_sigma
        sig2 = self.two_sigma
        args = (self.name+":", self.mean, sig1[0], sig1[1], sig2[0], sig2[1])
        return "<Parameter {:<15s} {:.4g} (+{:.4g} -{:.4g}) (+{:.4g} -{:.4g})>".format(*args)
        
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        return self.__repr__()
        
    #---------------------------------------------------------------------------
    @property
    def burnin(self):
        """
        The number of iterations to exclude as part of the "burn-in" phase
        """
        return self._burnin
    
    @burnin.setter
    def burnin(self, val):
        self._burnin = val
        del self.median, self.one_sigma, self.two_sigma
        
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
        Return the median of the trace, i.e., the 50th percentile of the trace
        """
        try:
            return self._median
        except AttributeError:
            self._median = np.percentile(self.flat_trace, 50.)
            return self._median
    
    @median.deleter
    def median(self):
        if hasattr(self, '_median'): delattr(self, '_median')
        
    #---------------------------------------------------------------------------
    @property
    def mean(self):
        """
        Return the "mean" as the the median, i.e., the 50th percentile of 
        the trace
        """
        return self.median
    
    #---------------------------------------------------------------------------
    @property
    def peak(self):
        """
        Return the value of the parameter that gives the peak of the 
        posterior PDF, as determined through Gaussian kernel density estimation
        """
        try:
            return self._peak
        except AttributeError:
            kern = scipy.stats.gaussian_kde(self.flat_trace)
            self._peak = scipy.optimize.fmin(lambda x: -kern(x), self.median, disp=False)[0]
            return self._peak
            
    @peak.deleter
    def peak(self):
        if hasattr(self, '_peak'): delattr(self, '_peak')
    
    #---------------------------------------------------------------------------
    @property
    def one_sigma(self):
        """
        Return the upper and lower one-sigma error intervals, as computed from
        the percentiles, `50 - 15.86555` and `84.13445 - 50`
        
        Returns
        -------
        upper, lower
            The lower and upper 1-sigma error intervals 
        """
        try: 
            return self._one_sigma
        except AttributeError:
            percentiles = [50., 15.86555, 84.13445]
            vals = np.percentile(self.flat_trace, percentiles)
            self._one_sigma = [vals[2] - vals[0], vals[0] - vals[1]]
            return self._one_sigma

    @one_sigma.deleter
    def one_sigma(self):
        if hasattr(self, '_one_sigma'): delattr(self, '_one_sigma')
    
    #---------------------------------------------------------------------------
    @property
    def two_sigma(self):
        """
        Return the upper and lower two-sigma error intervals, as computed from
        the percentiles, `50 - 2.2775` and `84.13445 - 50`

        Returns
        -------
        upper, lower
            The lower and upper 1-sigma error intervals 
        """
        try: 
            return self._two_sigma
        except AttributeError:
            percentiles = [50, 2.2775, 97.7225]
            vals = np.percentile(self.flat_trace, percentiles)
            self._two_sigma = [vals[2] - vals[0], vals[0] - vals[1]]
            return self._two_sigma
        
    @two_sigma.deleter
    def two_sigma(self):
        if hasattr(self, '_two_sigma'): delattr(self, '_two_sigma')
        
    #--------------------------------------------------------------------------- 
    def trace(self, niter=None):
        """
        Return the sample values at a specific iteration number.
        
        shape: (nwalkers, niters)
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
        
        # chain
        (walkers, _, ndim) = sampler.chain.shape
        inds = np.nonzero(sampler.chain)
        self.chain = sampler.chain[inds].reshape((walkers, -1, ndim)) # (nwalkers, niters, npars)
        
        # lnprob
        inds = np.nonzero(sampler.lnprobability)
        self.lnprobs = sampler.lnprobability[inds].reshape((self.walkers, -1)) # (nwalkers, niters)
        
        # other sampler attributes
        self.acceptance_fraction = sampler.acceptance_fraction
        self.autocorr_times = sampler.acor

        # make the constrained chain
        self._make_constrained_chain(fit_params)
        
        # make result params
        self._save_results()
        
        # set the burnin
        if burnin is None: 
            max_autocorr = 3*np.amax(self.autocorr_times)
            burnin = int(max_autocorr) if not np.isnan(max_autocorr) else int(0.1*self.iterations) 
            logger.info("setting the burnin period to {} iterations".format(burnin))
        self.burnin = int(burnin)
        
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        free_params = [self[name] for name in self.free_parameter_names]
        constrained_params = [self[name] for name in self.constrained_parameter_names]
        
        # first get the parameters
        toret = "Free parameters [ mean (+/-68%) (+/-95%) ]\n" + "_"*15 + "\n"
        toret += "\n".join(map(str, free_params))
        
        toret += "\n\nConstrained parameters [ mean (+/-68%) (+/-95%) ]\n" + "_"*22 + "\n"
        toret += "\n".join(map(str, constrained_params))
        
        return toret
    
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation
        """
        N = len(self.constrained_parameter_names)
        return "<EmceeResults: {} free paremeters, {} constrained parameters>".format(self.ndim, N)
        
    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        
        # check if key is the name of a free or constrained param
        if key in (self.free_parameter_names + self.constrained_parameter_names):
            return self._results[key]
        else:
            return getattr(self, key)
                    
    #---------------------------------------------------------------------------
    def __add__(self, other):
        """
        Add two `EmceeResults` objects together
        """
        if not isinstance(other, self.__class__):
            raise NotImplemented("Can only add two `EmceeResults` objects together")
        
        # check a few things first
        if sorted(self.free_parameter_names) != sorted(other.free_parameter_names):
            raise ValueError("Cannot add `EmceeResults` objects: mismatch in free parameters")
        if sorted(self.constrained_parameter_names) != sorted(other.constrained_parameter_names):
            raise ValueError("Cannot add `EmceeResults` objects: mismatch in constrained parameters")
        if self.walkers != other.chain.shape[0]:
            raise ValueError("Cannot add `EmceeResults` objects: mismatch in number of walkers")
        
        # copy to return
        toret = self.copy()
        
        # might need to reorder
        other_chain = other.chain
        if self.free_parameter_names != other.free_parameter_names:
            inds = [other.free_parameter_names.index(k) for k in self.free_parameter_names]
            other_chain = other_chain[...,inds]
            
        other_constrained_chain = other.constrained_chain
        if self.constrained_parameter_names != other.constrained_parameter_names:
            inds = [other.constrained_parameter_names.index(k) for k in self.constrained_parameter_names]
            other_constrained_chain = other_constrained_chain[...,inds]
        
        # add the chains together
        toret.chain = np.concatenate((self.chain, other_chain), axis=1)
        toret.constrained_chain = np.concatenate((self.constrained_chain, other_constrained_chain), axis=1)
        
        # add the log probs together
        toret.lnprobs = np.concatenate((self.lnprobs, other.lnprobs), axis=1)
        
        # update the new EmceeParameters
        toret._save_results()
        
        return toret
    
    #---------------------------------------------------------------------------
    def __radd__(self, other):
        return self.__add__(other)
            
    #---------------------------------------------------------------------------
    def copy(self):
        """
        Return a deep copy of the `EmceeResults` object
        """
        return copy.deepcopy(self)
            
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
            for nwalker, theta in enumerate(self.chain[:,niter,:]):
                
                # set the free parameters
                for val, name in zip(theta, self.free_parameter_names):
                    fit_params.set(name, val, update_constraints=False)

                # update constraints
                fit_params.update_constraints()
                                
                # set the constrained vals
                constrained_vals = fit_params.constrained_parameter_values
                self.constrained_chain[nwalker, niter, :] = constrained_vals
                
        # check for any constrained values that are fixed and remove
        tocat = ()
        names = []
        for i in range(shape[-1]):
            trace = self.constrained_chain[...,i]
            fixed = len(np.unique(trace)) == 1
            if not fixed:
                tocat += (trace[...,None],)
                names.append(self.constrained_parameter_names[i])
                
        self.constrained_parameter_names = names
        self.constrained_chain = np.concatenate(tocat, axis=2)
                
    #---------------------------------------------------------------------------
    def _save_results(self):
        """
        Make the dictionary of `EmceeParameters`
        """
        self._results = {}
        
        # the free parameters
        for i, name in enumerate(self.free_parameter_names):
            self._results[name] = EmceeParameter(name, self.chain[...,i])
        
        # the constrained parameters
        if self.constrained_chain is not None:
            for i, name in enumerate(self.constrained_parameter_names):
                self._results[name] = EmceeParameter(name, self.constrained_chain[...,i])
                        
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
        if val > self.iterations: val = 0
        self._burnin = val
        
        for param in self._results:
            self._results[param].burnin = val
    
    #---------------------------------------------------------------------------
    @property
    def max_lnprob(self):
        """
        The value of the maximum log probability
        """
        return np.amax(self.lnprobs)
        
    #---------------------------------------------------------------------------
    def max_lnprob_values(self, *names):
        """
        Return the value of the parameters at the iteration with the maximum
        probability
        """
        nwalker, niter = (self.lnprobs == self.max_lnprob).nonzero()
        nwalker, niter = nwalker[0], niter[0]
        return np.array([self[name].trace()[nwalker, niter] for name in names])
        
    #---------------------------------------------------------------------------
    def plot_timeline(self, *names, **kwargs):
        """
        Plot the chain timeline for as many parameters as specified in the 
        `names` tuple. This plots the value of each walker as a function
        of iteration
        
        Note: any iterations during the "burnin" period are excluded
        
        Parameters
        ----------
        names : tuple
            The string names of the parameters to plot
        outfile : str, optional
            If not `None`, save the resulting figure with the specified name
            
        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        """
        outfile = kwargs.get('outfile', None)
        N = len(names)
        if N < 1:
            raise ValueError('Must specify at least one parameter name for timeline plot')
            
        fig, axes = pfy.subplots(N, 1, sharex=True, figsize=(8, 9))
        if N == 1: axes = [axes]
        for i, name in enumerate(names):
            param = self[name]
            
            iter_num = range(self.iterations)[self.burnin:]
            trace = param.trace()[:, self.burnin:]
            
            # this excludes the burnin period
            for t in trace:
                axes[i].plot(iter_num, t, color="k", alpha=0.4)
            axes[i].yaxis.set_major_locator(MaxNLocator(5))
            axes[i].axhline(param.mean, color="#888888", lw=2)
            axes[i].set_ylabel(name)
        
        axes[-1].set_xlabel('iteration number', fontsize=16)
        if outfile is not None:
            fig.savefig(outfile)
            
        return fig
        
    #---------------------------------------------------------------------------
    def plot_triangle(self, *names, **kwargs):
        """
        Make the triange plot for as many parameters as specified in the 
        `names` tuple, optionally thinning the number of samples plotted.
        
        Note: any iterations during the "burnin" period are excluded
        
        Parameters
        ----------
        names : tuple
            The string names of the parameters to plot
        thin : int, optional
            The factor to thin the number of samples plotted by. Default is 
            1 (plot all samples)
        outfile : str, optional
            If not `None`, save the resulting figure with the specified name
            
        Returns
        -------
        fig : matplotlib.Figure
            The figure object 
        """
        thin = kwargs.get('thin', 1)
        outfile = kwargs.get('outfile', None)
        if len(names) < 2:
            raise ValueError('Must specify at least two parameter names to make triangle plot')
        
        # make the sample array for the desired parameters
        samples = []
        for name in names:
            param = self[name]
            trace = param.trace()[:, self.burnin::thin].flatten()
            samples.append(trace)
            
        fig = triangle.corner(np.vstack(samples).T, labels=names)
        if outfile is not None:
            fig.savefig(outfile)
        return fig
        
    #---------------------------------------------------------------------------
    def plot_2D_trace(self, param1, param2, outfile=None):
        """
        Plot the 2D traces of the given parameters, showing the 1 and 2 sigma
        contours
        
        Note: any iterations during the "burnin" period are excluded
        
        Parameters
        ----------
        param1 : str
            The name of the first parameter
        param2 : str
            The name of the second parameter
        outfile : str, optional
            If not `None`, save the resulting figure with the specified name
            
        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        
        """
        fig = pfy.figure()
        ax = fig.gca()
        
        names = self.free_parameter_names + self.constrained_parameter_names
        if not all(name in names for name in [param1, param2]):
            raise ValueError("Specified parameter names not valid")
            
        trace1 = self[param1].flat_trace
        trace2 = self[param2].flat_trace
        
        ax = tools.plot_mcmc_trace(ax, trace1, trace2, True, colors='k')
        ax.set_xlabel(param1, fontsize=16)
        ax.set_ylabel(param2, fontsize=16)
        
        # plot the "mean"
        ax.axvline(x=self[param1].mean, c="#888888", lw=1.5, alpha=0.4)
        ax.axhline(y=self[param2].mean, c="#888888", lw=1.5, alpha=0.4)
        
        if outfile is not None:
            fig.savefig(outfile)
        return fig

    #---------------------------------------------------------------------------
    def summarize_fit(self):
        """
        Summarize the fit, by plotting figures and outputing the relevant 
        information
        """
        args = (self.iterations, self.walkers, self.ndim)
        hdr = "Emcee fitting finished: {} iterations, {} walkers, {} free parameters\n".format(*args)
        
        logp = np.amax(self.lnprobs)
        chi2 = -2.*logp
        hdr += "Best log likelihood value = {:4f}, corresponding to chi2 = {:.4f}\n\n".format(logp, chi2)
        
        # print the results to the logger
        logger.info("\n"+hdr+str(self))
        
    #---------------------------------------------------------------------------
    def values(self):
        """
        Convenience function to return the values for the free parameters
        as an array
        """
        return np.array([self[name].mean for name in self.free_parameter_names])
    #---------------------------------------------------------------------------
#endclass EmceeResults

#-------------------------------------------------------------------------------
    
