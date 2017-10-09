from ... import numpy as np
from .. import logging
import scipy.stats
from collections import OrderedDict
import warnings

logger = logging.getLogger('rsdfit.emcee_results')
logger.addHandler(logging.NullHandler())

class EmceeParameter(object):
    """
    Class to hold the parameter fitting result
    """
    def __init__(self, name, trace, burnin=0):

        self.name   = name
        self._trace = trace # shape is (nwalkers, niters)
        self.burnin = int(burnin)

    def __repr__(self):
        sig1 = self.one_sigma
        sig2 = self.two_sigma
        args = (self.name+":", self.median, sig1[-1], sig1[0], sig2[-1], sig2[0])
        return "<Parameter {:<15s} {:.4g} (+{:.4g} {:.4g}) (+{:.4g} {:.4g})>".format(*args)

    def __str__(self):
        return self.__repr__()

    def _to_str(self, val):
        args = (self.name+":", val)
        return "<Parameter {:<15s} {:.4g}>".format(*args)


    @property
    def fiducial(self):
        """
        The fiducial value of the parameter
        """
        try:
            return self._fiducial
        except AttributeError:
            return None

    @fiducial.setter
    def fiducial(self, val):
        self._fiducial = val

    @property
    def burnin(self):
        """
        The number of iterations to exclude as part of the "burn-in" phase
        """
        return self._burnin

    @burnin.setter
    def burnin(self, val):
        self._burnin = val
        del self.median, self.one_sigma, self.two_sigma, self.three_sigma

    @property
    def error_rescaling(self):
        """
        Rescale errors by this factor
        """
        try:
            return self._error_rescaling
        except:
            return 1

    @error_rescaling.setter
    def error_rescaling(self, val):
        self._error_rescaling = val
        del self.one_sigma, self.two_sigma, self.three_sigma

    @property
    def flat_trace(self):
        """
        Returns the flattened chain, excluding steps that occured during the
        "burnin" period
        """
        return self._trace[:, self.burnin:].flatten()

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

    @property
    def mean(self):
        """
        Return the average value of the chain
        """
        return self.flat_trace.mean()

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

    @property
    def one_sigma(self):
        """
        Return the lower and upper one-sigma error intervals

        These are computed from the percentiles, ``50 - 15.86555`` and ``84.13445 - 50``

        Returns
        -------
        lower, upper
            The lower and upper 1-sigma error intervals
        """
        try:
            return self._one_sigma
        except AttributeError:
            percentiles = [50., 15.86555, 84.13445]
            vals = np.percentile(self.flat_trace, percentiles)*self.error_rescaling
            self._one_sigma = [-(vals[0] - vals[1]), vals[2] - vals[0]]
            return self._one_sigma

    @one_sigma.deleter
    def one_sigma(self):
        if hasattr(self, '_one_sigma'): delattr(self, '_one_sigma')

    @property
    def two_sigma(self):
        """
        Return the lower and upper two-sigma error intervals.

        These are computed from the percentiles, ``50 - 2.2775`` and ``97.7225 - 50``

        Returns
        -------
        lower, upper
            The lower and upper 2-sigma error intervals
        """
        try:
            return self._two_sigma
        except AttributeError:
            percentiles = [50, 2.2775, 97.7225]
            vals = np.percentile(self.flat_trace, percentiles)*self.error_rescaling
            self._two_sigma = [-(vals[0] - vals[1]), vals[2] - vals[0]]
            return self._two_sigma

    @two_sigma.deleter
    def two_sigma(self):
        if hasattr(self, '_two_sigma'): delattr(self, '_two_sigma')

    @property
    def three_sigma(self):
        """
        Return the lower and upper three-sigma error intervals

        There are computed from the percentiles, ``50 - 0.135`` and ``99.865 - 50``

        Returns
        -------
        lower, upper
            The lower and upper 3-sigma error intervals
        """
        try:
            return self._three_sigma
        except AttributeError:
            percentiles = [50, 0.135, 99.865]
            vals = np.percentile(self.flat_trace, percentiles)*self.error_rescaling
            self._three_sigma = [-(vals[0] - vals[1]), vals[2] - vals[0]]
            return self._three_sigma

    @three_sigma.deleter
    def three_sigma(self):
        if hasattr(self, '_three_sigma'): delattr(self, '_three_sigma')

    def trace(self, niter=None):
        """
        Return the sample values at a specific iteration number.

        shape: (nwalkers, niters)
        """
        if niter is None:
            return self._trace
        else:
            return self._trace[:,niter]

    @property
    def stderr(self):
        """
        The one-sigma standard error, averaging the lower and upper bounds
        """
        one_sigma = self.one_sigma
        return 0.5*(abs(one_sigma[0]) + one_sigma[1])


class EmceeResults(object):
    """
    Class to hold the fitting results from an `emcee` MCMC run
    """

    def __init__(self, sampler, fit_params, burnin=None, **meta):
        """
        Initialize with the `emcee` sampler and the fitting parameters
        """
        # store the parameter names
        self.free_names = fit_params.free_names
        self.constrained_names = fit_params.constrained_names

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
            self.autocorr_times = sampler.acor
        except:
            self.autocorr_times = np.zeros(len(self.free_names))

        # make the constrained chain
        self._make_constrained_chain(fit_params)

        # make result params
        self._save_results()

        # save fiducial values
        for name in self:
            self[name].fiducial = fit_params[name].fiducial

        # set the burnin
        if burnin is None:
            max_autocorr = 3*np.amax(self.autocorr_times)
            burnin = int(max_autocorr) if not np.isnan(max_autocorr) else int(0.1*self.iterations)
            logger.info("setting the burnin period to {} iterations".format(burnin))
        self.burnin = int(burnin)

        # save meta-data
        self.attrs = OrderedDict(**meta)

    def to_npz(self, filename, **meta):
        """
        Save the relevant information of the class to a numpy ``npz`` file
        """
        atts = ['free_names', 'constrained_names', 'chain', 'lnprobs',
                'acceptance_fraction', 'autocorr_times','constrained_chain',
                'burnin', 'attrs']

        self.attrs.update(**meta)
        d = {k:getattr(self, k) for k in atts}
        for k in ['model_version', 'pyrsd_version']:
            d[k] = getattr(self, k, None)
        np.savez(filename, **d)

    @classmethod
    def from_npz(cls, filename):
        """
        Load a numpy ``npz`` file and return the corresponding ``EmceeResults`` object
        """
        toret = cls.__new__(cls)
        with np.load(filename, encoding='latin1') as ff:
            for k, v in ff.items():
                if k == 'burnin' or k == 'attrs':
                    continue
                setattr(toret, k, v)

            toret._save_results()
            toret.burnin = int(ff['burnin'])

            if 'attrs' in ff:
                toret.attrs = ff['attrs'].tolist()
            else:
                toret.attrs = OrderedDict()

        toret.free_names = list(toret.free_names)
        toret.constrained_names = list(toret.constrained_names)
        return toret

    def __iter__(self):
        return iter(self.free_names + self.constrained_names)

    def to_str(self, max_lnprob=False):

        if not max_lnprob:
            return self.__str__()
        else:

            free_vals = self.max_lnprob_values()
            free_params = [self[name]._to_str(v) for name, v in zip(self.free_names, free_vals)]

            constrained_vals = self.max_lnprob_constrained_values()
            constrained_params = [self[name]._to_str(v) for name, v in zip(self.constrained_names, constrained_vals)]

            # first get the parameters
            toret = "Free parameters\n" + "_"*15 + "\n"
            toret += "\n".join(free_params)

            toret += "\n\nConstrained parameters\n" + "_"*22 + "\n"
            toret += "\n".join(constrained_params)
            return toret

    def __str__(self):
        free_params = [self[name] for name in self.free_names]
        constrained_params = [self[name] for name in self.constrained_names]

        # first get the parameters
        toret = "Free parameters [ median (+/-68%) (+/-95%) ]\n" + "_"*15 + "\n"
        toret += "\n".join(map(str, free_params))

        toret += "\n\nConstrained parameters [ median (+/-68%) (+/-95%) ]\n" + "_"*22 + "\n"
        toret += "\n".join(map(str, constrained_params))

        return toret

    def __repr__(self):
        N = len(self.constrained_names)
        return "<EmceeResults: {} free parameters, {} constrained parameters>".format(self.ndim, N)

    def __getitem__(self, key):

        # check if key is the name of a free or constrained param
        if key in (self.free_names + self.constrained_names):
            return self._results[key]
        else:
            return getattr(self, key)

    def verify_param_ordering(self, free_params, constrained_params):
        """
        Verify the ordering of `EmceeResults.chain`, making sure that the
        chains have the ordering specified by `free_params` and
        `constrained_params`
        """
        if sorted(self.free_names) != sorted(free_params):
            raise ValueError("mismatch in `EmceeResults` free parameters")
        if sorted(self.constrained_names) != sorted(constrained_params):
            raise ValueError("mismatch in `EmceeResults` constrained parameters")

        reordered = False
        # reorder `self.chain`
        if self.free_names != free_params:
            inds = [self.free_names.index(k) for k in free_params]
            self.chain = self.chain[...,inds]
            reordered = True

        # reorder self.constrained_chain
        if self.constrained_names != constrained_params:
            reordered = True

        if reordered:
            self.free_names = free_params
            self.constrained_names = constrained_params
            self._save_results()

    def __add__(self, other):
        """
        Add two `EmceeResults` objects together
        """
        if not isinstance(other, self.__class__):
            raise NotImplementedError("Can only add two `EmceeResults` objects together")

        # check a few things first
        if self.walkers != other.chain.shape[0]:
            raise ValueError("Cannot add `EmceeResults` objects: mismatch in number of walkers")

        # copy to return
        toret = self.copy()

        # verify the ordering of the other one
        other.verify_param_ordering(self.free_names, self.constrained_names)

        # add the chains together
        toret.chain = np.concatenate((self.chain, other.chain), axis=1)
        tmp = np.empty((toret.walkers, toret.iterations), dtype=self.constrained_chain.dtype)
        for name in tmp.dtype.names:
            tmp[name][:] = np.concatenate([self.constrained_chain[name], other.constrained_chain[name]], axis=1)[:]
        toret.constrained_chain = tmp

        # add the log probs together
        toret.lnprobs = np.concatenate((self.lnprobs, other.lnprobs), axis=1)

        # update the new EmceeParameters
        toret._save_results()
        return toret

    def __radd__(self, other):
        return self.__add__(other)

    def copy(self):
        """
        Return a deep copy of the `EmceeResults` object
        """
        import copy
        return copy.deepcopy(self)

    def _make_constrained_chain(self, fit_params):
        """
        Make the chain for the constrained parameters
        """
        if len(self.constrained_names) == 0:
            self.constrained_chain = None
            return

        # make the constrained chain from the other chain
        shape = (self.walkers, self.iterations)
        self.constrained_chain = np.empty(shape, dtype=fit_params.constrained_dtype)
        for niter in range(self.iterations):
            for nwalker, theta in enumerate(self.chain[:,niter,:]):

                # set the free parameters
                for val, name in zip(theta, self.free_names):
                    fit_params[name].value = val

                # update constraints
                fit_params.update_values()

                # set the constrained vals
                constrained_vals = fit_params.constrained_values
                for i, name in enumerate(self.constrained_names):
                    self.constrained_chain[name][nwalker, niter] = constrained_vals[i]

        # # check for any constrained values that are fixed and remove
        # tocat = ()
        # names = []
        # for i in range(shape[-1]):
        #     trace = self.constrained_chain[...,i]
        #     fixed = len(np.unique(trace)) == 1
        #     if not fixed:
        #         tocat += (trace[...,None],)
        #         names.append(self.constrained_names[i])
        #
        # self.constrained_names = names
        # self.constrained_chain = np.concatenate(tocat, axis=2)

    def _save_results(self):
        """
        Make the dictionary of `EmceeParameters`
        """
        self._results = {}

        # the free parameters
        for i, name in enumerate(self.free_names):
            self._results[name] = EmceeParameter(name, self.chain[...,i])

        # the constrained parameters
        if self.constrained_chain is not None:
            for i, name in enumerate(self.constrained_names):
                self._results[name] = EmceeParameter(name, self.constrained_chain[name])

    #---------------------------------------------------------------------------
    # some convenience attributes
    #---------------------------------------------------------------------------
    @property
    def iterations(self):
        """
        The number of iterations performed, as computed from the chain
        """
        return self.chain.shape[1]

    @property
    def walkers(self):
        """
        The number of walkers, as computed from the chain
        """
        return self.chain.shape[0]

    @property
    def ndim(self):
        """
        The number of free parameters
        """
        return len(self.free_names)

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

    @property
    def error_rescaling(self):
        """
        Rescale error on parameters due to covariance matrix from mocks
        """
        try:
            return self._error_rescaling
        except:
            return 1.

    @error_rescaling.setter
    def error_rescaling(self, val):
        self._error_rescaling = val

        for param in self._results:
            self._results[param].error_rescaling = val

    @property
    def max_lnprob(self):
        """
        The value of the maximum log probability
        """
        return self.lnprobs.max()

    def max_lnprob_values(self):
        """
        Return the value of the free parameters at the iteration with the maximum
        probability
        """
        nwalker, niter = np.unravel_index(self.lnprobs.argmax(), self.lnprobs.shape)
        return self.chain[nwalker, niter, :]

    def max_lnprob_constrained_values(self):
        """
        Return the value of the constrained parameters at the iteration
        with the maximum probability
        """
        nwalker, niter = np.unravel_index(self.lnprobs.argmax(), self.lnprobs.shape)
        return np.array(self.constrained_chain[nwalker, niter].tolist())

    def values(self):
        """
        Convenience function to return the median values for the free parameters
        as an array
        """
        return np.array([self[name].median for name in self.free_names])

    def constrained_values(self):
        """
        Convenience function to return the median values for the constrained
        parameters as an array
        """
        return np.array([self[name].median for name in self.constrained_names])

    def peak_values(self):
        """
        Convenience function to return the peak values for the free parameters
        as an array
        """
        return np.array([self[name].peak for name in self.free_names])

    def peak_constrained_values(self):
        """
        Convenience function to return the peak values for the
        constrained parameters
        as an array
        """
        return np.array([self[name].peak for name in self.constrained_names])

    def plot_timeline(self, *names, **kwargs):
        """
        Plot the chain timeline for as many parameters as specified in the
        `names` tuple.

        This plots the value of each walker as a function
        of iteration.

        Notes
        ------
        Any iterations during the "burnin" period are excluded

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
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from ..analysis import tex_names

        outfile = kwargs.get('outfile', None)
        N = len(names)
        if N < 1:
            raise ValueError('Must specify at least one parameter name for timeline plot')

        fig, axes = plt.subplots(N, 1, sharex=True, figsize=(8, 9))
        if N == 1: axes = [axes]
        for i, name in enumerate(names):
            param = self[name]

            iter_num = range(self.iterations)[self.burnin:]
            trace = param.trace()[:, self.burnin:]

            # this excludes the burnin period
            for t in trace:
                axes[i].plot(iter_num, t, color="k", alpha=0.4)
            axes[i].yaxis.set_major_locator(MaxNLocator(5))
            axes[i].axhline(param.median, color="#888888", lw=2)

            if name in tex_names:
                name = tex_names[name]
            axes[i].set_ylabel(name)

        axes[-1].set_xlabel('iteration number', fontsize=16)
        if outfile is not None:
            fig.savefig(outfile)

        return fig

    def plot_triangle(self, *names, **kwargs):
        """
        Make a triange plot for the desired parameters using the
        :func:`corner.corner` function.

        Notes
        -----
        Any iterations during the "burnin" period are excluded

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
        from ..analysis import tex_names

        try: import corner
        except: raise ImportError("`corner` must be installed")

        if len(names) < 2:
            raise ValueError('Must specify at least two parameter names to make triangle plot')

        thin = kwargs.pop('thin', 1)
        outfile = kwargs.pop('outfile', None)
        labels = [tex_names.get(name, name) for name in names]
        kwargs.setdefault('labels', labels)

        # make the sample array for the desired parameters
        samples = []
        for name in names:
            param = self[name]
            trace = param.trace()[:, self.burnin::thin].flatten()
            samples.append(trace)

        fig = corner.corner(np.vstack(samples).T, **kwargs)
        if outfile is not None:
            fig.savefig(outfile)
        return fig

    def jointplot_2d(self, param1, param2,
                        thin=1,
                        rename={},
                        crosshairs={},
                        **kwargs):
        """
        Plot the 2D traces of the given parameters, using KDE via the :func:`seaborn.jointplot` function.

        Notes
        -----
        Any iterations during the "burnin" period are excluded

        Parameters
        ----------
        param1 : str
            the name of the first parameter
        param2 : str
            the name of the second parameter
        thin : int, optional
            thin the plotted array randomly by this amount
        rename : dict, optional
            dictionary giving a string to rename each variable; default
            will try to use any latex names stored
        crosshairs : dict, optional
            values to show as horizontal or vertical lines
        **kwargs :
            additional keyword arguments to pass to ``seaborn``
        """
        import pandas as pd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import seaborn as sns
        from ..analysis import tex_names

        names = self.free_names + self.constrained_names
        if not all(name in names for name in [param1, param2]):
            raise ValueError("specified parameter names not valid")

        # default names
        rename.setdefault(param1, tex_names.get(param1, param1))
        rename.setdefault(param2, tex_names.get(param2, param2))

        # make the pandas Series of the flattened traces
        trace1 = self[param1].trace()[:, self.burnin::thin].flatten()
        trace1 = pd.Series(trace1, name=rename[param1])
        trace2 = self[param2].trace()[:, self.burnin::thin].flatten()
        trace2 = pd.Series(trace2, name=rename[param2])

        # do the plot
        kwargs.setdefault('space', 0)
        kwargs.setdefault('size', 7)
        with sns.axes_style("ticks"):
            g = sns.jointplot(trace1, trace2, kind="kde", **kwargs)

        # plot any cross-hairs
        ax = g.ax_joint
        if param1 in crosshairs:
            ax.axvline(x=crosshairs[param1], c="#888888", lw=1.5, alpha=0.4)
        if param2 in crosshairs:
            ax.axhline(y=crosshairs[param2], c="#888888", lw=1.5, alpha=0.4)

        return g


    def kdeplot_2d(self, param1, param2,
                        thin=1,
                        rename={},
                        crosshairs={},
                        **kwargs):
        """
        Plot the 2D traces of the given parameters, using KDE via :func:`seaborn.kdeplot`

        Notes
        -----
        Any iterations during the "burnin" period are excluded

        Parameters
        ----------
        param1 : str
            the name of the first parameter
        param2 : str
            the name of the second parameter
        thin : int, optional
            thin the plotted array randomly by this amount
        rename : dict, optional
            dictionary giving a string to rename each variable; default
            will try to use any latex names stored
        crosshairs : dict, optional
            values to show as horizontal or vertical lines
        **kwargs :
            additional keyword arguments to pass to ``seaborn``
        """
        import pandas as pd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import seaborn as sns
        from ..analysis import tex_names

        names = self.free_names + self.constrained_names
        if not all(name in names for name in [param1, param2]):
            raise ValueError("specified parameter names not valid")

        # default names
        rename.setdefault(param1, tex_names.get(param1, param1))
        rename.setdefault(param2, tex_names.get(param2, param2))

        # make the pandas Series of the flattened traces
        trace1 = self[param1].trace()[:, self.burnin::thin].flatten()
        trace1 = pd.Series(trace1, name=rename[param1])
        trace2 = self[param2].trace()[:, self.burnin::thin].flatten()
        trace2 = pd.Series(trace2, name=rename[param2])

        # do the plot
        kwargs.setdefault('shade', True)
        kwargs.setdefault('shade_lowest', False)
        with sns.axes_style("ticks"):
            ax = sns.kdeplot(trace1, trace2, **kwargs)

        # plot any cross-hairs
        if param1 in crosshairs:
            ax.axvline(x=crosshairs[param1], c="#888888", lw=1.5, alpha=0.4)
        if param2 in crosshairs:
            ax.axhline(y=crosshairs[param2], c="#888888", lw=1.5, alpha=0.4)

        return ax

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

    def as_dict(self, kind=None):
        """
        Return a dictionary of the values, either the `median`,
        `peak`, or `max_lnprob`
        """
        if kind is None or kind in ['mean', 'median']:
            funcs = [self.values, self.constrained_values]
        elif kind == 'peak':
            funcs = [self.peak_values, self.peak_constrained_values]
        elif kind == 'max_lnprob':
            funcs = [self.max_lnprob_values, self.max_lnprob_constrained_values]
        else:
            raise ValueError("`kind` must be one of ['mean', 'median', 'peak', 'max_lnprob']")

        toret = {}
        names = [self.free_names, self.constrained_names]
        for n, f in zip(names, funcs):
            d = dict(zip(n, f()))
            toret.update(d)

        return toret

    def chains(self, params=[], labels={}, use_latex=False):
        """
        Return a pandas DataFrame holding the chains (flat traces) as
        columns for both the free and constrained parameters

        Note that any burnin is removed from the chains.

        Parameters
        ----------
        params : list, {'free', 'constrained'}, optional
            return only a subset of the parameters
        use_latex : bool, optional
            If `True`, try to use any available latex names for the tick
            labels; default is `False`

        Returns
        -------
        DataFrame :
            the DataFrame holding the flat traces for all parameters
        """
        import pandas as pd
        from ..analysis import tex_names

        # do free or constrained
        if params == 'free':
            params = self.free_names
        elif params == 'constrained':
            params = self.constrained_names
        elif isinstance(params, str):
            params = [params]

        d = {}

        # add the free chains
        for p in self.free_names:
            if len(params) and p not in params:
                continue

            if p in labels:
                d[labels[p]] = self[p].flat_trace
            elif use_latex and p in tex_names:
                d[tex_names[p]] = self[p].flat_trace
            else:
                d[p] = self[p].flat_trace

        # add the constrained chains
        for p in self.constrained_names:
            if len(params) and p not in params:
                continue

            if p in labels:
                d[labels[p]] = self[p].flat_trace
            elif use_latex and p in tex_names:
                d[tex_names[p]] = self[p].flat_trace
            else:
                d[p] = self[p].flat_trace

        return pd.DataFrame(d)


    def corr(self, params=[], labels={}, use_latex=False):
        """
        Return a pandas DataFrame holding the correlation matrix,
        as computed from the ``chains`` DataFrame, for all
        parameters

        Parameters
        ----------
        params : list, {'free', 'constrained'}, optional
            return only a subset of the parameters
        use_latex : bool, optional
            If `True`, try to use any available latex names for the tick
            labels; default is `False`

        Returns
        -------
        DataFrame (Np, Np) :
            the DataFrame holding the correlation between parameters
        """
        return self.chains(params=params, labels=labels, use_latex=use_latex).corr()

    def sorted_1d_corrs(self, params=[], use_latex=False, absval=False):
        """
        Return a pandas DataFrame holding the correlation matrix,
        as computed from the ``chains`` DataFrame, for all
        parameters

        Parameters
        ----------
        params : list, {'free', 'constrained'}, optional
            return only a subset of the parameters
        use_latex : bool, optional
            If `True`, try to use any available latex names for the tick
            labels; default is `False`

        Returns
        -------
        DataFrame (Np, Np) :
            the DataFrame holding the correlation between parameters
        """
        # the 2D correlation matrix
        corr = self.corr(params=params, use_latex=use_latex)
        if absval: corr = corr.abs()

        # generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # set the upper triangle (including diag) to NaN
        corr.values[mask] = np.nan

        # unstack
        s = corr.unstack()

        # sort and remove the NaNs
        so = s.sort_values(kind="quicksort", ascending=False)
        so = so[~so.isnull()]

        # reset the index
        so = so.reset_index()
        so.columns = ['param_1', 'param_2', 'corr']
        return so


    def plot_correlation(self, params=[], use_latex=True, labelsize=10):
        """
        Plot the diagonal correlation matrix, using ``seaborn.heatmap``

        Parameters
        ----------
        params : list, {'free', 'constrained'}, optional
            return only a subset of the parameters
        use_latex : bool, optional
            If `True`, try to use any available latex names for the tick
            labels; default is `True`. You might want to set this to `False`
            if you are trying to use mpld3
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import seaborn as sns
        sns.set(style="white")
        f, ax = sns.plt.subplots(figsize=(11, 9))

        # Compute the correlation matrix
        corr = self.corr(params=params, use_latex=True)

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

        # format the ticks
        for f in [sns.plt.xticks, sns.plt.yticks]:
            locs, labels = f()
            sns.plt.setp(labels, rotation=45, fontsize=labelsize)

        return ax
