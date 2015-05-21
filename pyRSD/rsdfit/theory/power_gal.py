from ..parameters import Parameter, ParameterSet
from ... import rsd, numpy as np, os
import functools
import inspect
import copy
import tempfile

#-------------------------------------------------------------------------------
class GalaxyPowerParameters(ParameterSet):
    """
    A `ParameterSet` for the galaxy redshift space power spectrum in 
    `pyRSD.rsd.power_gal.GalaxySpectrum`
    """
    _model_params = {'sigma8': 'mass variance at r = 8 Mpc/h',
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
                     'sigma_sB': 'satB FOG damping in Mpc/h',
                     'small_scale_sigma': 'additional small scale velocity in km/s',
                     'N' : 'constant offset to model, in (Mpc/h)^3'}
                   
    _extra_params = {'b1_s': 'linear bias of satellites',
                     'b1_c': 'linear bias of centrals', 
                     'b1': 'the total linear bias', 
                     'fsigma8' : 'f(z)*sigma8(z) at z of measurement'}
                                      
    #---------------------------------------------------------------------------
    def __init__(self, filename, tag=None, extra_params=None):
        """
        Initialize by loading parameters from a file name
        
        Parameters
        ----------
        filename : str
            The name of the file to read the parameters from
        """
        # add in any extra parameters that we read
        if extra_params is not None:
            self._extra_params.update(extra_params)
            
        # initialize the base class
        super(GalaxyPowerParameters, self).__init__(filename, tag=tag, update_on_init=False)
                                        
        # add descriptions for the model params
        for name, desc in self._model_params.iteritems():
            if name in self and self[name] is not None:
                self.update_param(name, description=desc)
                
        # and the extra params
        for name, desc in self._extra_params.iteritems():
            if name in self and self[name] is not None:
                self.update_param(name, description=desc)
            else:
                self[name] = Parameter(name=name, description=desc)
            
    #---------------------------------------------------------------------------
    def __setitem__(self, name, value):
        """
        Only allow names to be set if they are a valid parameter, otherwise, 
        do nothing
        """
        if self.is_valid_parameter(name):
            ParameterSet.__setitem__(self, name, value)

    #---------------------------------------------------------------------------
    @property
    def valid_parameters(self):
        """
        Return the names of the valid parameters
        """
        try:
            return self._valid_parameters
        except AttributeError:
            self._valid_parameters = self._model_params.keys() + self._extra_params.keys()
            return self._valid_parameters
            
    #---------------------------------------------------------------------------
    def is_valid_parameter(self, name):
        """
        Check if the parameter name is valid
        """
        return name in self.valid_parameters
        
    #---------------------------------------------------------------------------
    def set_default_constraints(self):
        """
        Set the default constraints for this theory
        """       
        # central bias
        if 'b1_c' in self:
            self.add_constraint('b1_c', "(1 - fcB)*b1_cA + fcB*b1_cB", False)
        
        # satellite bias
        if 'b1_s' in self:
            self.add_constraint('b1_s', "(1 - fsB)*b1_sA + fsB*b1_sB", False)
        
        # total bias
        if 'b1' in self:
            self.add_constraint('b1', "(1 - fs)*b1_c + fs*b1_s", False)
        
    #---------------------------------------------------------------------------
    @property
    def model_params(self):
        """
        Return a dictionary of (name, value) for each name that is in 
        `GalaxyPowerParameters._model_params`
        """
        keys = self._model_params.keys()
        return dict((key, self[key].value) for key in keys if key in self)    
    #---------------------------------------------------------------------------   


#-------------------------------------------------------------------------------
class GalaxyPowerTheory(object):
    """
    Class representing a theory for computing the galaxy redshift space power
    spectrum. It handles the dependencies between model parameters and the 
    evaluation of the model itself.
    """
    
    def __init__(self, param_file, extra_param_file=None, k=None):
        """
        Initialize the theory 
        
        Parameters
        ----------
        param_file : str
            name of the file holding the parameters for the theory    
        extra_param_file : str
            name of the file holding the names of any extra parameter files
        k : array_like, optional
            If not `None`, initalize the theoretical model at these `k` values
        """
        # read the parameter file lines and save them for pickling   
        self._readlines = open(param_file, 'r').readlines()
         
        # read any extra parameters and make a dict
        if extra_param_file is not None:
            extra_params =  ParameterSet(extra_param_file)
            self.extra_params = extra_params.to_dict()
        else:
            self.extra_params = None
        
        # try to also read any extra params from the param file, tagged with 'theory_extra'
        extra_params = ParameterSet(param_file, tag='theory_extra')
        if len(extra_params) > 0:
            if self.extra_params is None:
                self.extra_params = extra_params.to_dict()
            else:
                self.extra_params.update(extra_params.to_dict())
        
        # read in the fit parameters; this should read only the keys that
        # are valid for the GalaxyPowerParameters
        self.fit_params = GalaxyPowerParameters(param_file, tag='theory', extra_params=self.extra_params)

        # read in the parameters again to get params that aren't fit params
        self.model_params = ParameterSet(param_file, tag='theory', update_on_init=False)
        
        # determine the theory callables
        self.pkmu_callable = self.model_params.get('pkmu_callable', 'Pgal')
        self.poles_callable = self.model_params.get('poles_callable', 'Pgal_poles')
        
        # now setup the model parameters; only the valid model kwargs are read
        allowable_model_params = rsd.power_gal.GalaxySpectrum.allowable_kwargs
        for param in self.model_params:
            if param not in allowable_model_params:
                del self.model_params[param] 
        
        # initialize the galaxy power spectrum model
        kwargs = {k:v() for k,v in self.model_params.iteritems()}
        if k is not None: kwargs['k'] = k
        self.model = rsd.power_gal.GalaxySpectrum(**kwargs)
        
        # update the constraints
        self.update_constraints()

    #---------------------------------------------------------------------------
    def to_file(self, filename, mode='w'):
        """
        Save the parameters of this theory in a file
        """            
        # first save the fit params
        self.fit_params.to_file(filename, mode=mode, header_name='theory params', 
                                footer=True, as_dict=True)
                                
        # now any extra params
        if self.extra_params is not None:
            f = open(filename, 'a')
            vals = ["theory_extra.%s =  %s" %(k, repr(v)) for k, v in self.extra_params.iteritems()]
            f.write("%s\n\n" %("\n".join(vals)))
            f.close()
        
        # now save the model params
        if self.pkmu_callable != 'Pgal':
            self.model_params.add('pkmu_callable', self.pkmu_callable)
        if self.poles_callable != 'Pgal_poles':
            self.model_params.add('poles_callable', self.poles_callable)       
        self.model_params.to_file(filename, mode='a', header_name='model params', 
                                  footer=True, as_dict=False)
        
    #---------------------------------------------------------------------------
    def update_constraints(self):
        """
        Update the constraints
        """
        # first add this to the symtable, before updating constraints
        self.fit_params._asteval.symtable['sigmav_from_bias'] = self.model.sigmav_from_bias
        
        # now do the constraints
        self.fit_params.set_default_constraints()
        self.set_model_dependent_constraints()
        
        # update
        self.fit_params.update_constraints()
        
    #---------------------------------------------------------------------------
    def set_model_dependent_constraints(self):
        """
        Setup any parameters that depend on the model
        """
        # f(z)*sigma8(z) at z of the model
        self.fit_params.add_constraint('fsigma8', "f*sigma8*%s" %self.model.D, False)
            
    #---------------------------------------------------------------------------
    # Properties
    #---------------------------------------------------------------------------
    @property
    def ndim(self):
        """
        Returns the number of free parameters, i.e., the `dimension` of the 
        theory
        """
        return len(self.free_parameter_names)
    
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
    def set_free_parameters(self, theta):
        """
        Given an array of values `theta`, set the free parameters of 
        `GalaxyPowerTheory.fit_params`. 
        
        Notes
        ------
        This assumes that theta is of the correct length, and sets the
        values in the same order as returned by `GalaxyPowerTheory.fit_params.free_parameters`
        """
        # set the parameters
        for val, name in zip(theta, self.free_parameter_names):
            self.fit_params.set(name, val, update_constraints=False)
           
        # only do this if the free params have finite priors
        if not np.isfinite(self.lnprior_free):
            return False
    
        # try to update
        try:
            self.fit_params.update_constraints()
        
            # also update the model 
            self.update_model()
        except:
            return False
        
        return True
        
    #---------------------------------------------------------------------------
    def model_callable(self, power_type, identifier, **kwargs):
        """
        Return the correct model function based on the type and identifier
        (from a PowerMeasurement)
        
        Any keyword arguments supplied will be passed to the callables
        """
        # computing pkmu
        if power_type == 'pkmu':
            if not hasattr(self.model, self.pkmu_callable):
                raise ValueError("RSD model has no function `%s` to compute P(k,mu)" %self.pkmu_callable)
                
            f = getattr(self.model, self.pkmu_callable)
            return functools.partial(f, np.array(identifier, ndmin=1), flatten=True, **kwargs)
        
        # computing multipoles
        elif power_type == 'pole':
            
            if not hasattr(self.model, self.poles_callable):
                raise ValueError("RSD model has no function `%s` to compute multipoles" %self.poles_callable)
                
            f = getattr(self.model, self.poles_callable)
            return functools.partial(f, np.array(identifier, ndmin=1), flatten=True, **kwargs)
            
        # all is lost...
        else:        
            # something has gone horribly wrong...
            msg = "No power spectrum model for measurement of type {} {}".format(power_type, identifier)
            raise NotImplementedError(msg)
    
    #---------------------------------------------------------------------------
    def update_model(self):
        """
        Update the theoretical model with the values currently in 
        `GalaxyPowerTheory.fit_params`
        """
        self.model.update(**self.fit_params.model_params)
    
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
            
    #---------------------------------------------------------------------------      

#-------------------------------------------------------------------------------