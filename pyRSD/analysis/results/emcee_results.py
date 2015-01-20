from ... import numpy as np, plotify as pfy
from . import triangle, tools

from matplotlib.ticker import MaxNLocator
import copy
import logging
import cPickle

logger = logging.getLogger('pyRSD.analysis.emcee_results')
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
        percentiles = [50., 15.86555, 84.13445]
        vals = np.percentile(self.flat_trace, percentiles)
        return vals[2] - vals[0], vals[0] - vals[1]

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
        percentiles = [50, 2.2775, 97.7225]
        vals = np.percentile(self.flat_trace, percentiles)
        return vals[2] - vals[0], vals[0] - vals[1]
    
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
        try:
            self.acor = sampler.acor
        except:
            self.acor = np.array([np.nan]*self.ndim)
        
        # make the constrained chain
        self._make_constrained_chain(fit_params)
        
        # make result params
        self._save_results()
        
        # set the burnin
        if burnin is None: burnin = int(0.1*self.iterations)
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
        if self.free_parameter_names != other.free_parameter_names:
            raise ValueError("Cannot add `EmceeResults` objects: mismatch in free parameters")
        if self.constrained_parameter_names != other.constrained_parameter_names:
            raise ValueError("Cannot add `EmceeResults` objects: mismatch in constrained parameters")
        if self.walkers != other.chain.shape[0]:
            raise ValueError("Cannot add `EmceeResults` objects: mismatch in number of walkers")
        
        # copy to return
        toret = self.copy()
        
        # add the chains together
        toret.chain = np.concatenate((self.chain, other.chain), axis=1)
        toret.constrained_chain = np.concatenate((self.constrained_chain, other.constrained_chain), axis=1)
        
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
                    fit_params[name].value = val
                                
                # set the constrained vals
                constrained_vals = np.array([fit_params[name].value for name in self.constrained_parameter_names])
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
        self._burnin = val
        
        for param in self._results:
            self._results[param].burnin = val
    
    #---------------------------------------------------------------------------
    def _plot_timeline(self, names):
        """
        Internal method to make plot timelines for a set of parameters
        """
        fig, axes = pfy.subplots(len(names), 1, sharex=True, figsize=(8, 9))

        # plot timeline for each free parameter
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
            
        return fig
            
    #---------------------------------------------------------------------------
    def plot_free_timelines(self, outfile=None):
        """
        Plot the timeline chains of the free parameters, excluding any 
        iterations in the burnin period
        """
        fig = self._plot_timeline(self.free_parameter_names)
        if outfile is not None:
            fig.savefig(outfile)
            
        return fig
        
    #---------------------------------------------------------------------------
    def plot_constrained_timelines(self, outfile=None):
        """
        Plot the timeline chains of the constrained parameters, excluding any 
        iterations in the burnin period
        """
        if len(self.constrained_parameter_names) > 0:
            fig = self._plot_timeline(self.constrained_parameter_names)
            if outfile is not None:
                fig.savefig(outfile)
            return fig
        
    #---------------------------------------------------------------------------
    def plot_free_triangle(self, outfile=None):
        """
        Make the triange plot for the free parameters
        """
        samples = self.chain[:, self.burnin:, :].reshape((-1, self.ndim))
        fig = triangle.corner(samples, labels=self.free_parameter_names)
        if outfile is not None:
            fig.savefig(outfile)
        return fig
        
    #---------------------------------------------------------------------------
    def plot_constrained_triangle(self, outfile=None):
        """
        Make the triange plot for the constrained parameters
        """
        N = len(self.constrained_parameter_names)
        samples = self.constrained_chain[:, self.burnin:, :].reshape((-1, N))
        fig = triangle.corner(samples, labels=self.constrained_parameter_names)
        if outfile is not None:
            fig.savefig(outfile)
        return fig
        
    #---------------------------------------------------------------------------
    def plot_2D_trace(self, param1, param2, outfile=None):
        """
        Plot the 2D traces of the given parameters, showing the 1 and 2 sigma
        contours
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
    def summarize_fit(self, label, make_plots=False):
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
        
        # now make some plots
        if make_plots:            
            self.plot_free_triangle(outfile=label+".free_triangle.pdf")
            self.plot_free_timelines(outfile=label+".free_timeline.pdf")
        
            self.plot_constrained_triangle(outfile=label+".constrained_triangle.pdf")
            self.plot_constrained_timelines(outfile=label+".constrained_timeline.pdf")
        
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
    
