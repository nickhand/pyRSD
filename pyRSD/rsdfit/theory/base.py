from pyRSD.rsdfit.parameters import Parameter, ParameterSet
from pyRSD.rsd._cache import Property
import numpy as np

from six import string_types
import contextlib
import functools
import warnings

def deprecated_parameter(func):
    """
    This is a decorator which can be used to mark parameters
    as deprecated
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("The model parameter is '%s' is deprecated" %func.__name__, category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)

    return new_func

class Schema(type):
    """
    Metaclass to gather all `free` and `fixed`
    attributes from the class
    """
    def __init__(cls, clsname, bases, attrs):

        # attach the registry attributes
        cls._free = set()
        cls._fixed = set()
        cls._model_params = set()
        cls._extra_params = set()
        cls._deprecated = set()

        # loop over each attribute
        for name in cls.__dict__:
            p = cls.__dict__[name]
            if isinstance(p, FreeProperty):
                if not p.deprecated:
                    cls._free.add(name)
                    if p.model_param: cls._model_params.add(name)
                    else: cls._extra_params.add(name)
                else:
                    cls._deprecated.add(name)
            elif isinstance(p, FixedProperty):
                if not p.deprecated:
                    cls._fixed.add(name)
                    if p.model_param: cls._model_params.add(name)
                    else: cls._extra_params.add(name)
                else:
                    cls._deprecated.add(name)


class FreeProperty(Property):
    """
    A free property
    """
    pass

class FixedProperty(Property):
    """
    A fixed property
    """
    pass


def free(model_param=True, deprecated=False):
    """
    Decorator to represent a freely varying model parameter
    """
    def dec(f):
        name = f.__name__
        _name = '__'+name

        @functools.wraps(f)
        def _get_property(self):
            val = f(self)
            val['vary'] = True
            val['name'] = name
            val['description'] = f.__doc__.strip()
            return Parameter(**val)

        if deprecated: _get_property = deprecated_parameter(_get_property)
        prop = FreeProperty(_get_property)
        prop.model_param = model_param
        prop.deprecated = deprecated
        return prop
    return dec

def fixed(model_param=False, deprecated=False):
    """
    Decorator to represent a model parameter, either fixed
    or free
    """
    def dec(f):
        name = f.__name__
        _name = '__'+name

        @functools.wraps(f)
        def _get_property(self):
            val = f(self)
            val['vary'] = False
            val['name'] = name
            val['description'] = f.__doc__.strip()
            return Parameter(**val)

        if deprecated: _get_property = deprecated_parameter(_get_property)
        prop = FixedProperty(_get_property)
        prop.model_param = model_param
        prop.deprecated = deprecated
        return prop
    return dec


class BasePowerParameters(ParameterSet):
    """
    A base `ParameterSet` class to represent parameters of a RSD model
    """
    defaults = None
    _model_cls = None

    @classmethod
    def from_defaults(cls, model=None, extra_params=[]):
        """
        Initialize from a default set of parameters

        Parameters
        ----------
        model : GalaxySpectrum, optional
            the model instance; if not provided, a new model
            will be initialized
        extra_params : list, optional
            list of names of extra parameters to be treated as valid
        """
        # initialize an empty class
        params = cls()

        # add extra parameters
        params.extra_params = []
        params.extra_params.extend(extra_params)

        # get the model
        if model is None:
            model = cls._model_cls()
        elif not isinstance(model, cls._model_cls):
            name = cls._model_cls.__name__
            raise TypeError("model should be a ``%s``" %name)

        # set the model
        params.model = model

        # delay asteval until all our loaded
        with params.delayed_asteval():

            # add the parameters
            params.add_many(*[par for par in cls.defaults])

            # first add this to the symtable, before updating constraints
            params.register_function('sigmav_from_bias', params.model.sigmav_from_bias)

        # update constraints
        params.prepare_params()
        params.update_constraints()

        return params

    @classmethod
    def from_file(cls, filename, model=None, tag=[], extra_params=[]):
        """
        Parameters
        ----------
        filename : str
            the name of the file to read the parameters from
        tag : str, optional
            only read the parameters with this label tag
        extra_params : list, optional
            a list of any additional parameters to treat as valid
        """
        # get the defaults first
        params = cls.from_defaults(model=model, extra_params=extra_params)

        # update descriptions
        with params.delayed_asteval():

            # update with values from file
            fromfile = super(BasePowerParameters, cls).from_file(filename, tags=tag)
            params.tag = fromfile.tag # transfer the tag
            for name in fromfile:

                # ignore deprecated parameters
                if name not in cls.defaults.deprecated:
                    params.add(**fromfile[name].to_dict())

            # first add this to the symtable, before updating constraints
            params.register_function('sigmav_from_bias', params.model.sigmav_from_bias)


        # update constraints
        params.prepare_params()
        params.update_constraints()

        # check for any fixed, constrained values
        for name in params:
            p = params[name]

            # if no dependencies are free, set vary = False, constrained = False
            if p.constrained:
                if not any(params[dep].vary for dep in p.deps):
                    p.vary = False
                    p.constrained = False

        return params

    @property
    def valid_model_params(self):
        """
        A list of the valid parameters names that can be passed
        to the ``GalaxySpectrum`` model instance
        """
        return self.defaults.model_params

    def __setitem__(self, name, value):
        """
        Only allow names to be set if they are a valid parameter, and
        if not, crash
        """
        if not self.is_valid(name):
            raise RuntimeError("`%s` is not a valid parameter name" %name)
        ParameterSet.__setitem__(self, name, value)

    def is_valid(self, name):
        """
        Check if the parameter name is valid
        """
        extras = getattr(self, 'extra_params', [])
        return name in self.defaults or name in self.defaults.deprecated or name in extras

    def to_dict(self):
        """
        Return a dictionary of (name, value) for each name that is in
        :attr:`model_params`
        """
        return dict((key, self[key].value) for key in self.valid_model_params if key in self)

    def set_free_parameters(self, theta):
        """
        Given an array of values `theta`, set the free parameters of
        `GalaxyPowerTheory.fit_params`

        Notes
        -----
        * if any free parameter values are outside their bounds, the
        model will not be updated and `False` will be returned

        Returns
        -------
        valid_model : bool
            return `True/False` flag indicating if we were able
            to successfully set the free parameters and update the model
        """
        # only set and update the model when all free params are
        # within max/min bounds and uniform prior bounds
        if not all(p.within_bounds(theta[i]) for i,p in enumerate(self.free)):
            return False

        # try to update
        try:
            self.update_values(**dict(zip(self.free_names, theta)))
        except Exception as e:
            import traceback
            msg = "exception while trying to update free parameters:\n"
            msg += "   current parameters:\n%s\n" %str(self.fit_params)
            msg += "   traceback:\n%s" %(traceback.format_exc())
            raise RuntimeError(msg)
        try:
            self.model.update(**self.to_dict())
        except Exception as e:
            import traceback
            msg = "exception while trying to update the theoretical model:\n"
            msg += "   current parameters:\n%s\n" %str(self)
            msg += "   traceback:\n%s" %(traceback.format_exc())
            raise RuntimeError(msg)

        return True

class BasePowerTheory(object):
    """
    A base class representing a theory for computing a redshift-space power
    spectrum.


    It handles the dependencies between model parameters and the
    evaluation of the model itself.
    """
    def __init__(self, model_cls, theory_cls, param_file,
                    model=None, extra_param_file=None, kmin=None, kmax=None):
        """
        Parameters
        ----------
        param_file : str
            name of the file holding the parameters for the theory
        extra_param_file : str
            name of the file holding the names of any extra parameter files
        model : subclass of , optional
            the model instance; if not provided, a new model
            will be initialized
        kmin : float, optional
            If not `None`, initalize the model with this `kmin` value
        kmax : float, optional
            If not `None`, initalize the model with this `kmax` value
        """
        # read in the parameters again to get params that aren't fit params
        self.model_params = ParameterSet.from_file(param_file, tags='model')

        # now setup the model parameters; only the valid model kwargs are read
        allowable_model_params = model_cls.allowable_kwargs
        for param in list(self.model_params.keys()):
            if param not in allowable_model_params and param != '__version__':
                del self.model_params[param]

        # store the kmin, kmax (used when setting model)
        self.kmin, self.kmax = kmin, kmax

        # set the model
        self._model_cls = model_cls
        self.model = model

        # read the parameter file lines and save them for pickling
        self._readlines = open(param_file, 'r').readlines()

        # read any extra parameters and make a dict
        self.extra_params = []
        if extra_param_file is not None:
            extra_params =  ParameterSet.from_file(extra_param_file)
            self.extra_params = extra_params.keys()

        # try to also read any extra params from the param file, tagged with 'theory_extra'
        extra_params = ParameterSet.from_file(param_file, tags='theory_extra')
        if len(extra_params):
            self.extra_params.extend(extra_params.keys())

        # read in the fit parameters; this should read only the keys that
        # are valid for the GalaxyPowerParameters
        kwargs = {'tag':'theory', 'model':self.model, 'extra_params':self.extra_params}
        self.fit_params = theory_cls.from_file(param_file, **kwargs)

        # delete any empty parameters
        for k in list(self.fit_params):
            if self.fit_params[k].value is None:
                del self.fit_params[k]

    def scale(self, theta):
        """
        Scale the (unscaled) free parameters, using the priors to
        define the scaling transformation
        """
        return (theta - self.fit_params.locs) / self.fit_params.scales

    def inverse_scale(self, theta):
        """
        Inverse scale the free parameters, using the priors to
        define the scaling transformation
        """
        return theta*self.fit_params.scales + self.fit_params.locs

    def scale_gradient(self, grad):
        """
        Scale the gradient with respect to the unscaled free parameters,
        using the priors to define the scaling transformation

        This returns df / dxprime where xprime is the scaled param vector
        """
        return grad * self.fit_params.scales

    @contextlib.contextmanager
    def preserve(self, theta):
        """
        Context manager that preserves the state of the model
        upon exiting the context by first saving and then restoring it
        """
        # save the free values
        original_state = self.free_values

        # set the input state
        for i, name in enumerate(self.free_names):
            self.fit_params[name].value = theta[i]

        yield

        # restore old state
        old = dict(zip(self.free_names, original_state))
        self.fit_params.update_values(**old)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        """
        Set the model, possibly from a file, or initialize a new one
        """
        from ..util import rsd_io

        # model kwargs
        kwargs = {k:v() for k,v in self.model_params.items()}
        if self.kmin is not None: kwargs['kmin'] = self.kmin
        if self.kmax is not None: kwargs['kmax'] = self.kmax

        if value is None:
            self._model = self._model_cls(**kwargs)
        elif isinstance(value, string_types):
            self._model = rsd_io.load_model(value, show_warning=False)
            self._model.update(**kwargs)
        elif isinstance(value, self._model_cls):
            self._model = value
            self._model.update(**kwargs)
        else:
            raise rsd_io.ConfigurationError("failure to set model in BasePowerTheory from file or instance")

        if not isinstance(self._model, self._model_cls):
            bad = self._model.__class__.__name__
            good = self._model_cls.__name__
            raise ValueError("model class is %s, but should be %s" % (bad, good))

        # set the fit params model too
        if hasattr(self, 'fit_params'):
            self.fit_params.model = self._model

    def to_file(self, filename, mode='w'):
        """
        Save the parameters of this theory in a file
        """
        # first save the fit params
        kwargs = {'mode':mode, 'header_name':'theory params', 'footer':True, 'as_dict':True}
        self.fit_params.to_file(filename, **kwargs)

        # now any extra params
        if self.extra_params is not None:
            f = open(filename, 'a')
            vals = []
            for name in self.extra_params:
                if name in self.fit_params:
                    desc = self.fit_params[name].description
                    vals.append("theory_extra.%s =  '%s'" %(name, desc))
            f.write("%s\n\n" %("\n".join(vals)))
            f.close()

        # now save the model params
        kwargs = {'mode':'a', 'header_name':'model params', 'footer':True, 'as_dict':False}
        self.model_params.to_file(filename, **kwargs)

    #---------------------------------------------------------------------------
    # properties
    #---------------------------------------------------------------------------
    @property
    def ndim(self):
        """
        Returns the number of free parameters, i.e., the `dimension` of the
        theory
        """
        return len(self.free_names)

    @property
    def lnprior_free(self):
        """
        Return the log prior for free parameters as the sum of the priors
        of each individual parameter
        """
        return sum(param.lnprior for param in self.free)

    @property
    def lnprior_constrained(self):
        """
        Return the log prior for constrained parameters as the sum of the priors
        of each individual parameter
        """
        return sum(param.lnprior for param in self.constrained)

    @property
    def lnprior(self):
        """
        Return the log prior for all "free" parameters as the sum of the priors
        of each individual parameter
        """
        return self.lnprior_free

    @property
    def dlnprior(self):
        """
        Return the derivative of the log prior for all "free" parameters
        """
        return np.array([p.dlnprior for p in self.free])

    #---------------------------------------------------------------------------
    # convenience attributes
    #---------------------------------------------------------------------------
    @property
    def free_fiducial(self):
        """
        Return an array of the fiducial free parameters
        """
        free = self.free_names
        params = self.fit_params
        toret = [params[key].fiducial for key in free]
        if None in toret:
            names = [free[i] for i in range(len(free)) if toret[i] is None]
            raise ValueError("fiducial values missing for parameters: %s" %str(names))
        return np.array(toret)

    @property
    def free_names(self):
        return self.fit_params.free_names

    @property
    def free_values(self):
        return self.fit_params.free_values

    @property
    def free(self):
        return self.fit_params.free

    @property
    def constrained_names(self):
        return self.fit_params.constrained_names

    @property
    def constrained_values(self):
        return self.fit_params.constrained_values

    @property
    def constrained(self):
        return self.fit_params.constrained

    #---------------------------------------------------------------------------
    # main functions
    #---------------------------------------------------------------------------
    def set_free_parameters(self, theta):
        """
        Given an array of values `theta`, set the free parameters of
        attr:`fit_params`

        Notes
        -----
        If any free parameter values are outside their bounds, the
        model will not be updated and `False` will be returned

        Returns
        -------
        valid_model : bool
            return `True/False` flag indicating if we were able
            to successfully set the free parameters and update the model
        """
        return self.fit_params.set_free_parameters(theta)

    def model_callable(self, data):
        """
        Return the correct model function based on the `PowerData`
        """
        import functools

        # if `data.transfer` is there, use it to evaluate power, accounting
        # for binning effects
        if data.transfer is not None and hasattr(self.model, 'from_transfer'):
            return functools.partial(self.model.from_transfer, data.transfer, flatten=True)

        # computing P(k,mu) with no binning effects
        if data.mode == 'pkmu':
            k = data.combined_k
            mu = data.combined_mu
            return functools.partial(self.model.power, k, mu)

        # computing P(k, ell) with no binning effects (i.e., integrating over P(k,mu))
        elif mode == 'poles':
            k = data.combined_k
            ell = data.combined_ell
            return functools.partial(self.model.poles, k, ell)

        # all is lost...
        # something has gone horribly wrong...
        raise NotImplementedError("failure trying to get model callable... all is lost")

    def check(self, return_errors=False):
        """
        Check the values of all parameters. Here, `check` means that
        each parameter is within its bounds and the prior is not infinity

        If `return_errors = True`, return the error messages as well
        """
        error_messages = []
        doing_okay = True

        # loop over each parameter
        for name in self.fit_params.free_names:
            par = self.fit_params[name]

            # check bounds
            if par.bounded and not par.within_bounds():
                doing_okay = False
                args = par.name, par.value, par.min, par.max
                msg = '{}={} is outside of reasonable limits [{}, {}]'.format(*args)
                error_messages.append(msg)
                continue

            # check prior
            if par.has_prior and np.isinf(par.lnprior):
                doing_okay = False
                msg = '{}={} is outside of prior {}'.format(par.name, par.value, par.prior)
                error_messages.append(msg)
                continue

        if return_errors:
            return doing_okay, error_messages
        else:
            return doing_okay
