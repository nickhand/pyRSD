"""
    parameter.py
    pyRSD.rsdfit.parameters

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : A class to store a generic Parameter, with a prior
"""

from . import tools, distributions as dists
from .. import lmfit
from ... import numpy as np
from ...rsd._cache import CachedModel, parameter, cached_property
import copy_reg

    
@CachedModel
class PickeableCache(object):
    
    def __new__(cls, *args, **kwargs):
        
        # make the new instance
        obj = object.__new__(cls, *args, **kwargs)
        
        # add the cache dictionary
        obj._cache = {}
        
        # register the pickling
        copy_reg.pickle(cls, _pickle, _unpickle)
        return obj

    def __init__(self):
        pass
        
def _pickle(param):
    d = {k:getattr(param, k, None) for k in param.keys()}
    d['prior'] = param.prior_name
    return _unpickle, (param.__class__, d, )

def _unpickle(cls, kwargs):
    return cls(**kwargs)

class Parameter(PickeableCache, lmfit.Parameter):
    """
    A subclass of ``lmfit.Parameter`` to represent a generic parameter. The added
    functionality is largely the ability to associate a prior with the parameter.
    Currently, the prior can either be a uniform or normal distribution.
    """
    valid_keys = ['name', 'value', 'vary', 'min', 'max', 'expr', 
                   'description', 'fiducial', 'prior', 'lower', 'upper', 
                   'mu', 'sigma', 'analytic']
                   
    def __init__(self, name=None, value=None, vary=False, min=None, max=None, expr=None, **kwargs):
        """
        Parameters
        ----------
        name : str
            The string name of the parameter. Must be supplied
        description : str, optional
            The string giving the parameter description
        value : object, optional
            The value. This can be a float, in which case the other attributes
            have meaning; otherwise the class just stores the object value.
            Default is `Parameter.fiducial`
        fiducial : object, optional
            A fiducial value to store for this parameter. Default is `None`.
        vary : bool, optional
            Whether to vary this parameter. Default is `False`.
        min : float, optional
            The minimum allowed limit for this parameter, for use in excluding
            "disallowed" parameter spaces. Default is `None`
        min : float, optional
            The maximum allowed limit for this parameter, for use in excluding
            "disallowed" parameter spaces. Default is `None`
        expr : {str, callable}, optional
            If a `str`, this gives a mathematical expression used to constrain 
            the value during the fit. Otherwise, the function will be called
            to evaluate the parameter's value. Default is `None`
        prior : {'uniform', 'normal'}, optional
            Use either a uniform or normal prior for this parameter. Default
            is no prior.
        lower, upper : floats, optional
            The bounds of the uniform prior to use
        mu, sigma : floats, optional
            The mean and std deviation of the normal prior to use
        """
        # set value to fiducial if it wasn't provided
        if value is None and kwargs.get('fiducial', None) is not None:
            value = kwargs['fiducial']
        lmfit.Parameter.__init__(self, name=name, value=value, vary=vary, min=min, max=max, expr=expr)
        self.value = self._val
        
        # handle additional parameters
        self.description = kwargs.get('description', 'No description available')
        self.prior_name = kwargs.get('prior', None)
        for k in ['fiducial', 'lower', 'upper', 'mu', 'sigma']:
            setattr(self, k, kwargs.get(k, None))
            
        # use analytic approximations for prior, bounds
        self.analytic = kwargs.get('analytic', False)
            
    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @property
    def output_value(self):
        """
        Explicit value for output -- defaults to `value`
        """
        try:
            return self._output_value
        except:
            return self.value
    
    @output_value.setter
    def output_value(self, val):
        self._output_value = val
    
    @parameter
    def analytic(self, val):
        """
        If `True`, use an analytic approximation for `Uniform` priors and the
        min/max bounds
        """
        self.min_bound.analytic = val
        self.max_bound.analytic = val
        if self.prior_name == 'uniform':
            self.prior.analytic = val
            
        return val
        
    @parameter
    def min(self, val):
        """
        The minimum allowed value (inclusive)
        """
        return val
        
    @parameter
    def max(self, val):
        """
        The maximum allowed value (exclusive)
        """
        return val
    
    @parameter
    def value(self, val):
        """
        The parameter value
        """
        
        if isinstance(val, str) and '$' in val:
            self.output_value = val
            val = tools.replace_vars(val, {})
        
        self.user_value = val
        self._val = val
        return self._getval()

    @parameter
    def fiducial(self, val):
        """
        A fiducial value associated with this parameter
        """
        return val
        
    @parameter
    def description(self, val):
        """
        The parameter description
        """
        return val
    
    @parameter
    def lower(self, val):
        """
        The lower limit of the uniform prior
        """
        return val
        
    @parameter
    def upper(self, val):
        """
        The upper limit of the uniform prior
        """
        return val
        
    @parameter
    def mu(self, val):
        """
        The mean value of the normal prior
        """
        return val
        
    @parameter
    def sigma(self, val):
        """
        The standard deviation of the normal prior
        """
        return val
                
    @parameter
    def prior_name(self, val):
        """
        The name of the prior distribution
        """
        allowed = ['uniform', 'normal', None]
        if val not in allowed:
            raise ValueError("Only allowed priors are %s" %allowed)
        return val
    
    @cached_property('prior_name', 'lower', 'upper', 'mu', 'sigma')
    def prior(self):
        """
        The prior distribution
        """
        if self.prior_name is None:
            return None
        elif self.prior_name == 'uniform':
            return dists.Uniform(self.lower, self.upper)
        elif self.prior_name == 'normal':
            return dists.Normal(self.mu, self.sigma)
        else:
            raise NotImplementedError("only `uniform` and `normal` priors currently implemented")    
    
    @cached_property('prior')
    def has_prior(self):
        """
        Whether the parameter has a prior defined
        """
        return self.prior is not None
        
    @cached_property('fiducial')
    def has_fiducial(self):
        """
        Whether the parameter has a fiducial value defined
        """
        return self.fiducial is not None
        
    @cached_property('min')
    def min_bound(self):
        """
        The distribution representing the minimum bound
        """
        return dists.MinimumBound(self.min)
        
    @cached_property('max')
    def max_bound(self):
        """
        The distribution representing the minimum bound
        """
        return dists.MaximumBound(self.max)
        
    @property
    def bounded(self):
        """
        Parameter is bounded if either `min` or `max` is defined
        """
        return self.min is not None or self.max is not None
    
    @property
    def constrained(self):
        """
        Parameter is constrained if the `Parameter.expr == None`
        """
        try:
            return self._constrained
        except:
            return self.expr is not None
        
    @constrained.setter
    def constrained(self, val):
        """
        Set the constrained value
        """
        self._constrained = val
            
    @property
    def lnprior(self):
        """
        Return log of the prior, which is either a uniform or normal prior.
        If the current value is outside `Parameter.min` or `Parameter.max`, 
        return `numpy.inf`
        """
        x = self.user_value
        
        # this will be 0 if within bounds, -np.inf otherwise
        lnprior = self.min_bound.log_pdf(x) + self.max_bound.log_pdf(x)
        
        # add in the log prior value (can also be -inf)
        if self.has_prior: 
            lnprior += self.prior.log_pdf(x)
   
        return lnprior
        
    @property
    def dlnprior(self):
        """
        Return the derivative of the log of the prior, which is either a 
        uniform or normal prior. If the current value is outside 
        `Parameter.min` or `Parameter.max`, return `numpy.inf`
        """
        x = self.user_value
        
        # this will be 0 if within bounds (and large pos/neg if out of bounds)
        dlnprior = self.min_bound.deriv_log_pdf(x) + self.max_bound.deriv_log_pdf(x)
        
        # add in the derivative of the log prior
        if self.has_prior:
            dlnprior += self.prior.deriv_log_pdf(x)
        
        return dlnprior
        
    @property
    def within_bounds(self):
        """
        Returns `True` if the specified value is within the (min, max)
        bounds and if the prior is uniform, within the lower/upper values
        of the prior
        """
        x = self.user_value
        toret = True
        if self.has_prior and self.prior_name == 'uniform':
            toret = (x >= self.prior.lower) and (x <= self.prior.upper)
        
        return toret and (x >= self.min_bound.value and x <= self.max_bound.value)
    
    #---------------------------------------------------------------------------
    # functions
    #---------------------------------------------------------------------------
    def __call__(self, output=False):
        if not output:
            return self.value
        else:
            return self.output_value

    def __repr__(self):
        s = []
        s.append("{0:<20}".format("'" + self.name + "'"))
        if tools.is_floatable(self.value):
            sval = "value={0:<15.5g}".format(self.value)
        else:
            sval = "value={0:<15s}".format(str(self.value) + ">")
            s.append(sval)

        if tools.is_floatable(self.value):
            fid_tag = ", fiducial" if self.value == self.fiducial else ""
            if self.constrained:
                t = " (constrained)"
            elif not self.vary:
                t = " (fixed%s)" %fid_tag
            else:
                t = " (free%s)" %fid_tag
            sval += "%-20s" %t

            s.append(sval)

            # add the prior
            if self.has_prior:
                s.append(str(self.prior))
            else:
                s.append('no prior')

        return "<Parameter %s" % ' '.join(s)

    def __str__(self):
        return self.__repr__()

    def keys(self):
        return self.valid_keys
      
    def get_value_from_prior(self, size=1):
        """
        Get random values from the prior, of size `size`
        """
        if not self.has_prior:
            print self
            raise ValueError("for parameter `%s`, cannot draw from prior that does not exist" %self.name)
            
        return self.prior.draw(size=size)
    
    def to_dict(self, output=False):
        """
        Convert the `Parameter` object to a dictionary
        """
        toret = {}
        for k in self.keys():
            if k == 'description': continue
            if k == 'prior':
                val = getattr(self, 'prior_name')
            elif k == 'value' and output:
                val = getattr(self, 'output_value')
            else:
                val = getattr(self, k)
        
            if val is not None:
                toret[k] = val
        
        for k in ['min', 'max']:
            if k in toret and not np.isfinite(toret[k]):
                toret.pop(k)
        return toret
