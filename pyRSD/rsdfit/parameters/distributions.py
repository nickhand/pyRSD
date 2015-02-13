from ... import numpy as np

import scipy.stats
from scipy.optimize import bisect
from scipy.interpolate import InterpolatedUnivariateSpline

# list of valid names and parameters for each distribution
valid_distributions = {'normal' : ['mu', 'sigma'], \
                       'uniform': ['lower', 'upper'], \
                       'trace' : ['trace'] }

def Distribution(name, **kwargs):
    """
    Return the distribution specified by `name`
    """
    if name not in valid_distributions:
        raise ValueError("Distribution name must be one of {0}".format(valid_distributions.keys()))
        
    if name == 'normal':
        return Normal(kwargs['mu'], kwargs['sigma'])
    elif name == 'uniform':
        return Uniform(kwargs['lower'], kwargs['upper'])
    elif name == 'trace':
        return Trace(kwargs['trace'])
        
#-------------------------------------------------------------------------------
        
def histogram_bins(signal):
    """
    Return optimal number of bins.
    """
    select = np.logical_not(np.isnan(signal)) & np.logical_not(np.isinf(signal))
    h = (3.5 * np.std(signal[select])) / (len(signal[select])**(1./3.))
    bins = int(np.ceil((max(signal[select])-min(signal[select])) / h))*2

    return bins
                           
#-------------------------------------------------------------------------------
class DistributionBase(object):
    """
    Base class for representing a `Distribution`
    """
    def __init__(self, name, **params):
        """
        Initialize a distribution.
        
        Parameters
        ----------
        name : str 
            the name of the distribution; must be in `valid_distributions`
        params : dict
            the distribution parameters as a dictionary
        """
        # make the name is valid
        self.name = name.lower()
        if self.name not in valid_distributions.keys():
            args = self.name, valid_distributions.keys()
            raise ValueError("Name '{0}' not valid; must be one of {1}".format(*args))
        
        # make sure we have the right parameters
        if any(k not in valid_distributions[self.name] for k in params):
            raise ValueError("Incorrect parameters for distribution of type {0}".format(self.name))
        
        # add the specific distribution parameters
        self.params = valid_distributions[self.name]
        for k, v in params.iteritems():
            setattr(self, k, v)

    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        name = self.name.title()
        pars = ", ".join(['{}={}'.format(key, getattr(self, key)) for key in sorted(self.params)])
        return "{}({})".format(name, pars)
        
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
        return self.__str__()
        
    #---------------------------------------------------------------------------
    def update(self, **params):
        """
        Update the distribution parameters.
        """
        for k, v in params.iteritems():
            if not hasattr(self, k):
                raise AttributeError("Cannot update attribute '{0}'; no such attribute".format(k))
            setattr(self, k, v)        
    #---------------------------------------------------------------------------
#endclass DistributionBase

#-------------------------------------------------------------------------------
class Normal(DistributionBase):
    """
    The Normal distribution.
    
    The normal distribution has two parameters in its `params` attribute:
    
        - `mu` (:math:`\mu`): the location parameter (the mean)
        - `sigma` (:math:`\sigma`): the scale parameter (standard deviation)
    
    Example usage::
    
        >>> norm = Normal(5.0, 1.0)
        >>> print(norm)
        Normal(mu=5.0, sigma=1.0)
        
    The limits for this distribution are by default defined as the 3 sigma
    limits:
    
        >>> print(norm.limits())
        (2.0, 8.0)
    
    Draw 5 random values from the distribution::
    
        >>> print(norm.draw(5))
        [ 3.98397393  5.25521791  4.221863    5.14080982  5.33466166]
    
    Get a grid of values, equidistantly sampled in probability:
    
        >>> print(norm.grid(11))
        [ 6.38299413  5.96742157  5.67448975  5.4307273   5.21042839  5.
          4.78957161  4.5692727   4.32551025  4.03257843  3.61700587]        
        
    """
    def __init__(self, mu, sigma):
        """
        Initiate a normal distribution.
        
        Parameters
        -----------
        mu : float
            location parameter (mean)
        sigma : float
            scale parameter (standard deviation)
        """
        super(Normal, self).__init__('normal', mu=mu, sigma=sigma)
    
    #---------------------------------------------------------------------------
    def limits(self, factor=1.):
        """
        Return the minimum and maximum of the normal distribution.
        
        For a normal distribution, we define the upper and lower limit as
        
        ..math: \pm 3*factor*self.sigma
        """
        lower = self.mu - 3*self.sigma*factor
        upper = self.mu + 3*self.sigma*factor
        return lower, upper
    
    #---------------------------------------------------------------------------
    def draw(self, size=1):
        """
        Draw random value(s) from the normal distribution.
        
        Parameters
        ----------
        size : int 
            the number of values to generate
    
        Returns
        -------
        values : float, array_like 
            random value(s) from the distribution
        """
        kwargs = {'size' : size, 'loc' : self.loc, 'scale' : self.scale}
        values = np.random.normal(**kwargs)
        return values if size > 1 else values[0]
    
    #---------------------------------------------------------------------------
    def grid(self, sampling=5):
        """
        Draw regularly sampled values from the distribution, spaced 
        according to the probability density.
        
        We grid uniformly in the cumulative probability density function.
        """
        cum_sampling = np.linspace(0., 1., sampling+2)[1:-1]
        mygrid = scipy.stats.norm(loc=self.mu, scale=self.sigma).isf(cum_sampling)
        return mygrid
    
    #---------------------------------------------------------------------------
    def pdf(self, domain=None, factor=1.):
        """
        Return the probability distribution function
        
        Parameters
        ----------
        domain : array_like
            The values to compute the pdf at. If `None`, sample equidistantly
            between the limits of the distrbution
        factor : float
            The factor by which to scale the distribution limits
        """
        # If 'domain' is not given, sample equidistantly between the limits
        # of the distribution
        if domain is None:
            lower, upper = self.limits(factor=factor)
            domain = np.linspace(lower, upper, 1000)
          
        kwargs = {'loc' : self.loc, 'scale' : self.scale}
        return domain, getattr(scipy.stats.norm(**kwargs), 'pdf')(domain)
        
    #---------------------------------------------------------------------------
    def cdf(self, domain=None, factor=1.):
        """
        Return the cumulative distribution function

        Parameters
        ----------
        domain : array_like
            The values to compute the cdf at. If `None`, sample equidistantly
            between the limits of the distrbution
        factor : float
            The factor by which to scale the distribution limits
        """
        # If 'domain' is not given, sample equidistantly between the limits
        # of the distribution
        if domain is None:
            lower, upper = self.limits(factor=factor)
            domain = np.linspace(lower, upper, 1000)

        kwargs = {'loc' : self.loc, 'scale' : self.scale}
        return domain, getattr(scipy.stats.norm(**kwargs), 'cdf')(domain)
        
    #---------------------------------------------------------------------------
    def shrink(self, factor=10.):
        """
        Shrink the extent of the normal distribution.
        
        Parameters
        ----------
        factor : float
            the factor with which to shrink the distribution
        """
        self.sigma /= factor
        
    #---------------------------------------------------------------------------
    def expand(self, factor=10.):
        """
        Expand the extent of the normal distribution.
        
        Parameters
        ----------
        factor : float
            the factor with which to expand the distribution
        """
        self.sigma *= factor
    
    #---------------------------------------------------------------------------   
    def plot(self, on_axis='x', **kwargs):
        """
        Plot a normal prior to the current axes.
        """
        import plotify as pfy
        
        ax = kwargs.pop('ax', pfy.gca())
        lower, upper = self.limits()
        domain = np.linspace(lower, upper, 500)
        domain, mypdf = self.pdf(domain=domain)
        if on_axis.lower() == 'y':
            domain, mypdf = mypdf, domain
        ax.plot(domain, mypdf, **kwargs)
    
    #---------------------------------------------------------------------------
    @property
    def loc(self):
        """
        Return the `loc` parameter used by `numpy`, equal here to `self.mu`
        """
        return self.mu
    
    #---------------------------------------------------------------------------
    @property
    def scale(self):
        """
        Return the `scale` parameter used by `numpy`, equal here to `self.sigma`
        """
        return self.sigma
    #---------------------------------------------------------------------------
    
#endclass Normal

#-------------------------------------------------------------------------------
class Uniform(DistributionBase):
    """
    The Uniform distribution.
    """
    def __init__(self, lower, upper):
        """
        Initiate a normal distribution.
        
        Parameters
        -----------
        lower : float
            the lower limit of the distribution
        upper : float
            the upper limit of the distribution
        """
        super(Uniform, self).__init__('uniform', lower=lower, upper=upper)
    
    #---------------------------------------------------------------------------
    @property
    def lower(self):
        """
        Make this a property in case the supplied `lower` is actually a function
        """
        if callable(self._lower):
            return self._lower()
        else:
            return self._lower
            
    @lower.setter
    def lower(self, val):
        self._lower = val
        
    #---------------------------------------------------------------------------
    @property
    def upper(self):
        """
        Make this a property in case the supplied `upper` is actually a function
        """
        if callable(self._upper):
            return self._upper()
        else:
            return self._upper
            
    @upper.setter
    def upper(self, val):
        self._upper = val
        
    #---------------------------------------------------------------------------
    def limits(self, factor=1.):
        """
        Return the minimum and maximum of the uniform distribution.
        """
        lower = self.lower
        upper = self.upper
        window = upper - lower
        mean = 0.5*(lower + upper)
        lower = mean - 0.5*factor*window
        upper = mean + 0.5*factor*window  
        
        return lower, upper
    
    #---------------------------------------------------------------------------
    def draw(self, size=1):
        """
        Draw random value(s) from the uniform distribution.
        
        Parameters
        ----------
        size : int 
            the number of values to generate
    
        Returns
        -------
        values : float, array_like 
            random value(s) from the distribution
        """
        kwargs = {'size' : size, 'low' : self.lower, 'high' : self.upper}
        values = np.random.uniform(**kwargs)
        return values if size > 1 else values[0]

    #---------------------------------------------------------------------------
    def grid(self, sampling=5):
        """
        Draw a (set of) likely value(s) from the uniform distribution.
        """
        return np.linspace(self.lower, self.upper, sampling)
        
    #---------------------------------------------------------------------------
    def pdf(self, domain=None, factor=1.):
        """
        Return the probability distribution function
        
        Parameters
        ----------
        domain : array_like
            The values to compute the pdf at. If `None`, sample equidistantly
            between the limits of the distrbution
        factor : float
            The factor by which to scale the distribution limits
        """
        # If 'domain' is not given, sample equidistantly between the limits
        # of the distribution
        if domain is None:
            lower, upper = self.limits(factor=factor)
            domain = np.linspace(lower, upper, 1000)
          
        kwargs = {'loc' : self.loc, 'scale' : self.scale}
        return domain, getattr(scipy.stats.uniform(**kwargs), 'pdf')(domain)
        
    #---------------------------------------------------------------------------
    def cdf(self, domain=None, factor=1.):
        """
        Return the cumulative distribution function

        Parameters
        ----------
        domain : array_like
            The values to compute the cdf at. If `None`, sample equidistantly
            between the limits of the distrbution
        factor : float
            The factor by which to scale the distribution limits
        """
        # If 'domain' is not given, sample equidistantly between the limits
        # of the distribution
        if domain is None:
            lower, upper = self.limits(factor=factor)
            domain = np.linspace(lower, upper, 1000)

        kwargs = {'loc' : self.loc, 'scale' : self.scale}
        return domain, getattr(scipy.stats.norm(**kwargs), 'cdf')(domain)

    #---------------------------------------------------------------------------
    def plot(self, on_axis='x', **kwargs):
        """
        Plot a uniform prior to the current axes.
        """
        import plotify as pfy
        
        ax = kwargs.pop('ax', pfy.gca())
        lower, upper = self.limits()
        if on_axis.lower() == 'x':
            kwargs['y'] = 1
            kwargs['xmin'], kwargs['xmax'] = self.limits()
            ax.axhline(**kwargs)
        elif on_axis.lower() =='y':
            kwargs['x'] = 1
            kwargs['ymin'], kwargs['ymax'] = self.limits()
            ax.axyline(**kwargs)
        
    #---------------------------------------------------------------------------
    @property
    def loc(self):
        """
        Return the `loc` parameter used by `numpy`, equal here to `self.lower`
        """
        return self.lower
    
    #---------------------------------------------------------------------------
    @property
    def scale(self):
        """
        Return the `scale` parameter used by `numpy`, equal here to 
        `self.upper` - `self.lower`
        """
        return self.upper - self.lower
    #---------------------------------------------------------------------------
#endclass Uniform

#-------------------------------------------------------------------------------
class Trace(DistributionBase):
    """
    A distribution from a MCMC trace.
    
    The Trace Distribution has one parameter in its `params` attribute:
    
        - `trace`: the trace array
    
    Example usage::
    
        >>> trace = Trace(np.random.normal(size=10))
        >>> print(trace)
        Trace(...)
        
    """
    #---------------------------------------------------------------------------
    def __init__(self, trace):
        """
        Initiate a trace distribution.
        
        Parameters
        -----------
        trace : array_like
            the array holding the trace samples from a MCMC sampler
        """
        super(Trace, self).__init__('trace', trace=trace)
    
    #---------------------------------------------------------------------------
    def limits(self, factor=1.):
        """
        Return the minimum and maximum of the trace distribution.
        """
        median = np.median(self.trace)
        minim = np.min(self.trace)
        maxim = np.max(self.trace)
        
        lower = median - factor * (median-minim)
        upper = median + factor * (maxim-median)
        
        return lower, upper
    
    #---------------------------------------------------------------------------
    def draw(self, size=1):
        """
        Draw a (set of) random value(s) from the trace.
    
        Parameters
        ----------
        size : int, optional
            the number of random_variates to return; default = 1
        """
        return self._draw_from_discrete(size=size)
    
    #---------------------------------------------------------------------------
    def grid(self, sampling=5):
        """
        Draw a (set of) likely value(s) from the trace
        
        """
        cum_sampling = np.linspace(0., 1., sampling+2)[1:-1]
        return self._draw_from_discrete(size=sampling, cum_sampling=cum_sampling)
    
    #---------------------------------------------------------------------------
    def _draw_from_discrete(self, size=1, cum_sampling=None):
        """
        Returns a random variate from the discrete distribution using the
        inverse transform sampling method

        Parameters
        ----------
        size : int, optional
            the number of random_variates to return; default = 1
        cum_sampling : array_like
            values between 0 and 1 to sample the cumulative distribution at. If
            `None`, draw random cumulative samples
        """
        # get the binned cdf
        domain, dist = self.cdf()
        F = 1.*dist/dist.max()

        spline = InterpolatedUnivariateSpline(domain, F)
        xmin = np.amin(domain)    
        xmax = np.amax(domain)     
        
        # now draw randoms and get the values
        if cum_sampling is None:
            rands = np.random.random(size=size)
        else:
            rands = cum_sampling
        toret = np.array([bisect(lambda a: spline(a)-r, xmin, xmax) for r in rands])
        
        return toret if size > 1 else toret[0]
    #end _draw_from_discrete
    
    #---------------------------------------------------------------------------
    def pdf(self, domain=None, factor=1.):
        """
        Return the (binned) probability distribution function
        
        Parameters
        ----------
        domain : array_like
            The values to compute the pdf at. If `None`, sample equidistantly
            between the limits of the distrbution
        factor : float
            The factor by which to scale the distribution limits
        """
        keep = np.logical_not(np.isnan(self.trace)) & np.logical_not(np.isinf(self.trace))
        bins = histogram_bins(self.trace[keep])
        counts, domain_ = np.histogram(self.trace[keep], bins=bins, density=True)
        domain_ = 0.5*(domain_[:-1] + domain_[1:])
        
        if domain is not None:
            counts = np.interp(domain, domain_, counts)
        else:
            domain = domain_
        return domain, counts
        
    #---------------------------------------------------------------------------
    def cdf(self, domain=None, factor=1.):
        """
        Return the (binned) cumulative distribution function

        Parameters
        ----------
        domain : array_like
            The values to compute the cdf at. If `None`, sample equidistantly
            between the limits of the distrbution
        factor : float
            The factor by which to scale the distribution limits
        """
        keep = np.logical_not(np.isnan(self.trace)) & np.logical_not(np.isinf(self.trace))
        bins = histogram_bins(self.trace[keep])
        counts, domain_ = np.histogram(self.trace[keep], bins=bins, density=True)
        cdf = np.cumsum(np.diff(domain_) * counts)
        domain_ = 0.5*(domain_[:-1] + domain_[1:])
        if domain is not None:
            cdf = np.interp(domain, domain_, cdf)
        else:
            domain = domain_
        return domain, cdf

    #---------------------------------------------------------------------------
    @property
    def loc(self):
        """
        Return the `loc` parameter used by `numpy`, equal here to the
        median of `self.trace`
        """
        keep = np.logical_not(np.isnan(self.trace)) & np.logical_not(np.isinf(self.trace))
        return np.median(self.trace[keep])
    
    #---------------------------------------------------------------------------
    @property
    def scale(self):
        """
        Return the `scale` parameter used by `numpy`, equal here to the standard
        deviation of `self.trace`
        """
        keep = np.logical_not(np.isnan(self.trace)) & np.logical_not(np.isinf(self.trace))
        return np.std(self.trace[keep])
    
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        old_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=8)
        name = self.name.title()
        pars = ", ".join(['{}={}'.format(key, getattr(self, key)) for key in sorted(self.params)])
        np.set_printoptions(threshold=old_threshold) 
        return "{}({})".format(name, pars)
    
    #---------------------------------------------------------------------------
    def plot(self, on_axis='x', **kwargs):
        """
        Plot the trace distribution to the current axes.
        """
        import plotify as pfy
        
        ax = kwargs.pop('ax', pfy.gca())
        domain, pdf = self.pdf()
        ax.plot(domain, pdf, **kwargs)
    #---------------------------------------------------------------------------
#endclass Trace

#-------------------------------------------------------------------------------