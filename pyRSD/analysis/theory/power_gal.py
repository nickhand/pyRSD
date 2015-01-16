from ..parameters import Parameter, ParameterSet
from ... import rsd, numpy as np, os
import functools
import inspect
import copy
import tempfile

#-------------------------------------------------------------------------------
def deriving_function(func, params, param_name):
    """
    Function to be used to derive parameter values from other parameters
    """
    return func(params[param_name]())

#-------------------------------------------------------------------------------
class GalaxyPowerParameters(ParameterSet):
    """
    A `ParameterSet` for the galaxy redshift space power spectrum in 
    `pyRSD.rsd.power_gal.GalaxySpectrum`
    """
    _valid_keys = {'sigma8': 'mass variance at r = 8 Mpc/h',
                   'f': 'growth rate, f = dlnD/dlna', 
                   'alpha_perp': 'perpendicular Alcock-Paczynski effect parameter', 
                   'alpha_par': 'parallel Alcock-Paczynski effect parameter', 
                   'b1_sA': 'linear bias of sats w/o other sats in same halo',
                   'b1_sB': 'linear bias of sats w/ other sats in same halo',
                   'b1_cA': 'linear bias of cens w/o sats in same halo',
                   'b1_cB': 'linear bias of cens w/ sats in same halo',
                   'fcB': 'fraction of cens with sats in same halo',
                   'fsB': 'fraction of sats with other sats in same halo', 
                   'fs': 'fraction of total galaxies that are satellites',
                   'NcBs': 'amplitude of 1-halo power for cenB-sat in (Mpc/h)^3', 
                   'NsBsB': 'amplitude of 1-halo power for satB-satB in (Mpc/h)^3', 
                   'sigma_c': 'centrals FOG damping in Mpc/h',
                   'sigma_s': 'satellite FOG damping in Mpc/h',
                   'sigma_sA': 'satA FOG damping in Mpc/h', 
                   'sigma_sB': 'satB FOG damping in Mpc/h'}
                   
    _extra_keys = {'b1_s': 'linear bias of satellites',
                   'b1': 'the total linear bias', 
                   'b1_c': 'linear bias of centrals', 
                   'log10_NcBs' : 'log10 of NcBs', 
                   'log10_NsBsB' : 'log10 of NsBsB',
                   'fsigma8' : 'f(z)*sigma8(z) at z of measurement'}
                   
    def __init__(self, *args):
        """
        Initialize, optionally loading parameters from a file name
        """
        # initialize the base class
        super(GalaxyPowerParameters, self).__init__()
                
        # now possibly load from file
        loaded = False
        if (len(args) == 1 and isinstance(args[0], basestring)):
            loaded = True
            self.load(args[0])
            
        # load the valid_keys first
        for name, desc in self._valid_keys.iteritems():
            if name in self and self[name] is not None:
                self.update_param(name, description=desc)
            else:
                self[name] = Parameter(name=name, description=desc)
                
        # and the extras
        for name, desc in self._extra_keys.iteritems():
            if name in self and self[name] is not None:
                self.update_param(name, description=desc)
            else:
                self[name] = Parameter(name=name, description=desc)

        # setup the dependencies
        if loaded: self._set_default_constraints()
            
    #---------------------------------------------------------------------------
    def __setitem__(self, key, value):
        """
        Only allow names to be set if they are in `self._valid_keys` or 
        `self._extra_keys`. If not, do nothing
        """
        if self._is_valid_key(key):
            ParameterSet.__setitem__(self, key, value)
    
    #---------------------------------------------------------------------------
    def _is_valid_key(self, key):
        """
        Check if the key is valid
        """
        return key in self._valid_keys or key in self._extra_keys
        
    #---------------------------------------------------------------------------
    def _set_default_constraints(self):
        """
        Set the default constraints for this theory
        """       
        # central bias
        self.add_constraint('b1_c', "(1 - {fcB})*{b1_cA} + {fcB}*{b1_cB}")
        
        # satellite bias
        self.add_constraint('b1_s', "(1 - {fsB})*{b1_sA} + {fsB}*{b1_sB}")
        
        # total bias
        self.add_constraint('b1', "(1 - {fs})*{b1_c} + {fs}*{b1_s}")
        
        # log of constants
        self.add_constraint('log10_NcBs', "numpy.log10({NcBs})")
        self.add_constraint('log10_NsBsB', "numpy.log10({NsBsB})")
        
    #---------------------------------------------------------------------------
    @property
    def model_params(self):
        """
        Return a dictionary of (name, value) for each name that is in 
        `self._valid_keys`
        """
        keys = self._valid_keys.keys()
        return dict((key, self[key].value) for key in keys)
    
    #---------------------------------------------------------------------------
    @property
    def free_parameter_names(self):
        """
        Only return the free parameter names in `self._valid_keys`. This is 
        useful for setting parameters of the model that have changed.
        """
        names = ParameterSet.free_parameter_names.fget(self)
        return [name for name in names if name in self._valid_keys]
    
    #---------------------------------------------------------------------------
    @property
    def constrained_parameter_names(self):
        """
        Only return the constrained parameter names in `self._valid_keys`. This 
        is useful for setting parameters of the model that have changed.
        """
        names = ParameterSet.constrained_parameter_names.fget(self)
        return [name for name in names if name in self._extra_keys]
    
    #---------------------------------------------------------------------------
    
#endclass GalaxyPowerParameters

#-------------------------------------------------------------------------------
class GalaxyPowerTheory(object):
    """
    Class representing a theory for computing the galaxy redshift space power
    spectrum. It handles the dependencies between model parameters and the 
    evaluation of the model itself.
    """
    
    def __init__(self, param_file, k=None):
        """
        Setup the theory 
        
        Parameters
        ----------
        param_file : str
            name of the file holding the parameters for the theory
        k : array_like, optional
            If not `None`, initalize the theoretical model at these `k` values
        """   
        self._init_args = (param_file, k)
         
        # read in the fit parameters; this should extra only the keys that
        # are valid for the GalaxyPowerParameters
        self.fit_params = GalaxyPowerParameters(param_file)

        # now setup the model parameters; params are those not in `fit_params`
        self.model_params = ParameterSet(param_file, params_only=True)
        for param in self.model_params:
            if self.fit_params._is_valid_key(param):
                del self.model_params[param] 
        
        # initialize the galaxy power spectrum model
        kwargs = {k:v() for k,v in self.model_params.iteritems() if k in rsd.power_gal.GalaxySpectrum.allowable_kwargs}
        if k is not None: kwargs['k'] = k
        self.model = rsd.power_gal.GalaxySpectrum(**kwargs)
        
        # setup any params depending on model
        self._set_model_dependent_params()
                
    #---------------------------------------------------------------------------
    def __getstate__(self):
        lines = open(self._init_args[0], 'r').readlines()
        return {'lines' : lines, 'model' : self.model}
    
    def __setstate__(self, d):
        f = tempfile.NamedTemporaryFile(delete=False)
        name = f.name
        lines = "\n".join(d['lines'])
        f.write(lines)
        f.close()
    
        # now set up the object
        self.fit_params = GalaxyPowerParameters(name)

        # now setup the model parameters; params are those not in `fit_params`
        self.model_params = ParameterSet(name, params_only=True)
        for param in self.model_params:
            if self.fit_params._is_valid_key(param):
                del self.model_params[param] 
    
        self.model = d['model']
        
        # setup any params depending on model
        self._set_model_dependent_params()
        
        if os.path.exists(name):
            os.remove(name)
    
    #---------------------------------------------------------------------------
    def _set_model_dependent_params(self):
        """
        Setup any parameters that depend on the model
        """
        # f*sigma8 at z of the model
        self.fit_params.add_constraint('fsigma8', "{f}*{sigma8}*%s" %self.model.D)
        
        # relation between sigma and bias
        r = self.model.sigma_bias_relation
        
        # sat sigma
        f = functools.partial(deriving_function, r.sigma, self.fit_params, 'b1_s')
        self.fit_params['sigma_s'].deriving_function = f
        
        # satA sigma
        f = functools.partial(deriving_function, r.sigma, self.fit_params, 'b1_sA')
        self.fit_params['sigma_sA'].deriving_function = f
        
        # satB sigma
        f = functools.partial(deriving_function, r.sigma, self.fit_params, 'b1_sB')
        self.fit_params['sigma_sB'].deriving_function = f
        
        # sat bias
        f = functools.partial(deriving_function, r.bias, self.fit_params, 'sigma_s')
        self.fit_params['b1_s'].deriving_function = f
        
        # satA bias
        f = functools.partial(deriving_function, r.bias, self.fit_params, 'sigma_sA')
        self.fit_params['b1_sA'].deriving_function = f
        
        # satB bias
        f = functools.partial(deriving_function, r.bias, self.fit_params, 'sigma_sB')
        self.fit_params['b1_sB'].deriving_function = f
            
    #---------------------------------------------------------------------------
    @property
    def lnprior_free(self):
        """
        Return the log prior for free parameters as the sum of the priors
        of each individual parameter
        """       
        return sum(param.lnprior for param in self.free_parameters)
        
    #---------------------------------------------------------------------------
    @property
    def lnprior_constrained(self):
        """
        Return the log prior for constrained parameters as the sum of the priors
        of each individual parameter
        """       
        return sum(param.lnprior for param in self.constrained_parameters)
        
    #---------------------------------------------------------------------------
    @property
    def lnprior(self):
        """
        Return the log prior for all "changed" parameters as the sum of the priors
        of each individual parameter
        """       
        return self.lnprior_free + self.lnprior_constrained
        
    #---------------------------------------------------------------------------
    def set_free_parameters(self, theta):
        """
        Given an array of values `theta`, set the free parameters of 
        `self.fit_params`. 
        
        Note: This assumes that theta is of the correct length, and sets the
        values in the same order as returned by `self.fit_params.free_parameters`
        """
        # set the parameters
        for val, name in zip(theta, self.free_parameter_names):
            self.fit_params[name].value = val
            
        # also update the model 
        self.update_model()
            
    #---------------------------------------------------------------------------
    @property
    def free_parameter_names(self):
        """
        Convenience property
        """
        return self.fit_params.free_parameter_names

    #---------------------------------------------------------------------------
    @property
    def free_parameter_values(self):
        """
        Convenience property
        """
        return self.fit_params.free_parameter_values

    #---------------------------------------------------------------------------
    @property
    def free_parameters(self):
        """
        Convenience property
        """
        return self.fit_params.free_parameters

    #---------------------------------------------------------------------------
    @property
    def constrained_parameter_names(self):
        """
        Convenience property
        """
        return self.fit_params.constrained_parameter_names

    #---------------------------------------------------------------------------
    @property
    def constrained_parameter_values(self):
        """
        Convenience property
        """
        return self.fit_params.constrained_parameter_values

    #---------------------------------------------------------------------------
    @property
    def constrained_parameters(self):
        """
        Convenience property
        """
        return self.fit_params.constrained_parameters

    #---------------------------------------------------------------------------
    def model_callable(self, power_type, identifier, **kwargs):
        """
        Return the correct model function based on the type and identifier
        (from a PowerMeasurement)
        
        Any keyword arguments supplied will be passed to the callables
        """
        if power_type == 'pkmu':
            return functools.partial(self.model.Pgal, np.array(identifier), flatten=True, **kwargs)
        elif power_type == 'pole':
            if identifier == 0:
                if len(kwargs) == 0:
                    return self.model.Pgal_mono
                else:
                    return functools.partial(self.model.Pgal_mono, *kwargs)
            elif identifier == 2:
                if len(kwargs) == 0:
                    return self.model.Pgal_quad
                else:
                    return functools.partial(self.model.Pgal_quad, *kwargs)
                
        # if we get here, we messed up
        msg = "No power spectrum model for measurement of type".format(power_type, identifier)
        raise NotImplementedError(msg)
    
    #---------------------------------------------------------------------------
    def update_model(self):
        """
        Update the theoretical model with the values currently in 
        `self.fit_params`
        """
        self.model.update(**self.fit_params.model_params)
    
    #---------------------------------------------------------------------------
    @property
    def ndim(self):
        """
        Returns the number of free parameters, i.e., the `dimension` of the 
        theory
        """
        return len(self.free_parameter_names)
    
    #---------------------------------------------------------------------------
    def check(self, return_errors=False):
        """
        Check the values of all parameters. Here, `check` means that
        each parameter is within its bounds and the prior is not infinity
        
        If `return_errors = True`, return the error messages as well
        """
        error_messages = []
        doing_okay = True
        
        # loop over each parameter
        for name in self.fit_params:
            par = self.fit_params[name]

            # check bounds
            if par.bounded and not par.within_bounds:
                doing_okay = False
                args = par.name, par.value, par.limits
                msg = '{}={} is outside of reasonable limits {}'.format(*args)
                error_messages.append(msg)
                continue
                
            # check prior
            if par.has_prior() and np.isinf(par.lnprior): 
                doing_okay = False
                msg = '{}={} is outside of prior {}'.format(par.name, par.value, par.prior)
                error_messages.append(msg)
                continue
        
        if return_errors:
            return doing_okay, error_messages
        else:
            return doing_okay   
            
    #---------------------------------------------------------------------------      
#endclass GalaxyPowerTheory

#-------------------------------------------------------------------------------