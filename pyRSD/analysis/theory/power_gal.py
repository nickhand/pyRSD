from .parameters import Parameter, ParameterSet
from .. import rsd
import functools

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
                   'log10_NsBsB' : 'log10 of NsBsB'}
                   
    def __init__(self, *args):
        """
        Initialize, optionally loading parameters from a file name
        """
        # initialize the base class
        super(GalaxyPowerParameters, self).__init__(self)
                
        # now possibly load from file
        loaded = False
        if (len(args) == 1 and isinstance(args[0], basestring)):
            loaded = True
            self.load(args[0])
            
        # load the valid_keys first
        for name, desc in self._valid_keys.iteritems():
            if name in self:
                self.update_param(name, description=desc)
            else:
                self[name] = Parameter(name=name, description=desc)
                
        # and the extras
        for name, desc in self._extra_keys.iteritems():
            if name in self:
                self.update_param(name, description=desc)
            else:
                self[name] = Parameter(name=name, description=desc)

        # setup the dependencies
        if loaded: self._set_default_constraints()
    
    #---------------------------------------------------------------------------
    def __setitem__(self, key, value):
        """
        Only allow names to be set if they are in `self._valid_keys`
        """
        if key not in self._valid_keys and key not in self._extra_keys:
            raise KeyError("Key '%s' is not a valid key for `GalaxyPowerParameters`" %key)
        ParameterSet.__setitem__(self, key, value)
    
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
#endclass GalaxyPowerParameters

#-------------------------------------------------------------------------------
class GalaxyPowerTheory(object):
    """
    Class representing a theory for computing the galaxy redshift space power
    spectrum. It handles the dependencies between model parameters and the 
    evaluation of the model itself.
    """
    
    def __init__(self, model_params_file=None, fit_params_file=None):
        
        # read in the parameters used to initialize the model
        if model_params_file is not None:
            self.model_params = ParameterSet(model_params_file)
        else:
            self.model_params = ParameterSet()
        
        # read in the fit parameters
        if fit_params_file is not None:
            self.fit_params = GalaxyPowerParameters(fit_params_file)
        else:
            self.fit_params = GalaxyPowerParameters()
        
        # initialize the galaxy power spectrum model
        kwargs = {k:v() for k,v in self.model_params.iteritems() if k in power_gal.GalaxySpectrum.allowable_kwargs}
        self.model = rsd.power_gal.GalaxySpectrum(**kwargs)
        
        # setup any derived parameters
        self._setup_derived_params()
        
    #---------------------------------------------------------------------------
    def _setup_derived_params(self):
        """
        Setup the deriving functions for parameters that can be derived
        """
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
    def lnprior(self):
        """
        Return the log prior for all free parameters as the sum of the priors
        of each individual parameter
        
        Note: free parameters are defined as those parameters with `vary = True`
        and `constrained = False`
        """       
        return sum(param.lnprior for param in self.free_parameters)
        
    #---------------------------------------------------------------------------
    def set_free_parameters(self, theta):
        """
        Given an array of values `theta`, set the free parameters of 
        `self.fit_params`. 
        
        Note: This assumes that theta is of the correct length, and sets the
        values in the same order as returned by `self.fit_params.free_parameters`
        """
        for val, name in zip(theta, self.free_parameter_names):
            self.fit_params[name].value = val
    
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
    