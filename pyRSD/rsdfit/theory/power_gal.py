"""
    power_gal.py
    pyRSD.rsdfit.theory

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : A ParameterSet for the rsd.GalaxySpectrum class
"""
from . import base_model_params, extra_model_params
from ..parameters import Parameter, ParameterSet
from ... import rsd, numpy as np, os

import contextlib

class GalaxyPowerParameters(ParameterSet):
    """
    A `ParameterSet` for the galaxy redshift space power spectrum in 
    `pyRSD.rsd.GalaxySpectrum`
    """
    model_params = base_model_params
    extra_params = extra_model_params
         
    @classmethod                 
    def from_file(cls, filename, tag=None, extra_params={}):
        """        
        Parameters
        ----------
        filename : str
            the name of the file to read the parameters from
        tags : str, optional
            only read the parameters with this label tag
        extra_params : dict, optional
            a dictionary (name, description) of any additional parameters
        """  
        # add in any extra parameters that we read
        cls.extra_params.update(extra_params)          
        
        # initialize the base class
        params = super(GalaxyPowerParameters, cls).from_file(filename, tags=tag)
                                   
        # add descriptions for the model params
        for name, desc in cls.model_params.items():
            if name in params and params[name] is not None:
                params.update_param(name, description=desc)
                
        # and the extra params
        for name, desc in cls.extra_params.items():
            if name in params and params[name] is not None:
                params.update_param(name, description=desc)
            else:
                params[name] = Parameter(name=name, description=desc)
        
        params.model_params = cls.model_params
        params.extra_params = cls.extra_params
        return params
            
    def __setitem__(self, name, value):
        """
        Only allow names to be set if they are a valid parameter, and
        if not, crash
        """
        if not self.is_valid(name):
            raise RuntimeError("`%s` is not a valid GalaxyPowerParameters parameter name" %name)
        ParameterSet.__setitem__(self, name, value)
            
    @property
    def valid_parameters(self):
        """
        Return the names of the valid parameters
        """
        return list(self.model_params.keys()) + list(self.extra_params.keys())
            
    def is_valid(self, name):
        """
        Check if the parameter name is valid
        """
        return name in self.valid_parameters
        
    def set_default_constraints(self):
        """
        Set the default constraints for this theory
        """       
        # central bias
        if 'b1_c' in self:
            self['b1_c'].expr = "(1 - fcB)*b1_cA + fcB*b1_cB"
        
        # satellite bias
        if 'b1_s' in self:
            self['b1_s'].expr = "(1 - fsB)*b1_sA + fsB*b1_sB"
        
        # total bias
        if 'b1' in self:
            self['b1'].expr = "(1 - fs)*b1_c + fs*b1_s"
        
        # f*sigma8 at z
        if 'f' in self and 'sigma8_z' in self:
            self['fsigma8'].expr = "f*sigma8_z" 
            
        # b1*sigma8 at z
        if 'b1' in self and 'sigma8_z' in self:
            self['b1sigma8'].expr = "b1*sigma8_z"
            
        # AP related
        if 'alpha_perp' in self and 'alpha_par' in self:
            self['F_AP'].expr    = "alpha_par/alpha_perp"
            self['alpha'].expr   = "(alpha_perp**2 * alpha_par)**(1./3)"
            self['epsilon'].expr = "(alpha_perp/alpha_par)**(-1./3) - 1.0"
            
    def to_dict(self):
        """
        Return a dictionary of (name, value) for each name that is in 
        `GalaxyPowerParameters.model_params`
        """
        keys = self.model_params.keys()
        return dict((key, self[key].value) for key in keys if key in self)    


class GalaxyPowerTheory(object):
    """
    A class representing a theory for computing the galaxy redshift space power
    spectrum. It handles the dependencies between model parameters and the 
    evaluation of the model itself.
    """
    
    def __init__(self, param_file, extra_param_file=None, kmin=None, kmax=None):
        """        
        Parameters
        ----------
        param_file : str
            name of the file holding the parameters for the theory    
        extra_param_file : str
            name of the file holding the names of any extra parameter files
        kmin : float, optional
            If not `None`, initalize the model with this `kmin` value
        kmax : float, optional
            If not `None`, initalize the model with this `kmax` value
        """
        # read the parameter file lines and save them for pickling   
        self._readlines = open(param_file, 'r').readlines()
         
        # read any extra parameters and make a dict
        self.extra_params = {}
        if extra_param_file is not None:
            extra_params =  ParameterSet.from_file(extra_param_file)
            self.extra_params = extra_params.to_dict()
        
        # try to also read any extra params from the param file, tagged with 'theory_extra'
        extra_params = ParameterSet.from_file(param_file, tags='theory_extra')
        if len(extra_params):
            self.extra_params.update(extra_params.to_dict())
        
        # read in the fit parameters; this should read only the keys that
        # are valid for the GalaxyPowerParameters
        kwargs = {'tag':'theory', 'extra_params':self.extra_params}
        self.fit_params = GalaxyPowerParameters.from_file(param_file, **kwargs)

        # read in the parameters again to get params that aren't fit params
        self.model_params = ParameterSet.from_file(param_file, tags='model')

        # now setup the model parameters; only the valid model kwargs are read
        allowable_model_params = rsd.GalaxySpectrum.allowable_kwargs
        for param in self.model_params.keys():
            if param not in allowable_model_params and param != '__version__':
                del self.model_params[param] 
        
        # store the kmin, kmax
        self.kmin, self.kmax = kmin, kmax
                        
        # set the model
        self.set_model()
        
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
                
    def set_model(self, *model):
        """
        Set the model, possibly from a file, or initialize a new one
        """
        from ..util import rsd_io
        
        # model kwargs
        kwargs = {k:v() for k,v in self.model_params.items()}
        if self.kmin is not None: kwargs['kmin'] = self.kmin
        if self.kmax is not None: kwargs['kmax'] = self.kmax
    
        if not len(model):
            self.model = rsd.GalaxySpectrum(**kwargs)
        elif isinstance(model[0], str):
            self.model = rsd_io.load_model(model[0])
            self.model.update(**kwargs)
        elif isinstance(model[0], rsd.GalaxySpectrum):
            self.model = model[0]
            self.model.update(**kwargs)
        else:
            raise rsd_io.ConfigurationError("failure to set model in `GalaxyPowerTheory` from file or instance")
    
        # update the constraints
        self.update_constraints()    
    
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
            vals = ["theory_extra.%s =  %s" %(k, repr(v)) for k, v in self.extra_params.items()]
            f.write("%s\n\n" %("\n".join(vals)))
            f.close()
        
        # now save the model params
        kwargs = {'mode':'a', 'header_name':'model params', 'footer':True, 'as_dict':False}     
        self.model_params.to_file(filename, **kwargs)
        
    def update_constraints(self):
        """
        Update the constraints
        """
        # first add this to the symtable, before updating constraints
        self.fit_params.register_function('sigmav_from_bias', self.model.sigmav_from_bias)
        
        # now do the constraints
        self.fit_params.set_default_constraints()
        
        # update
        self.fit_params.prepare_params()
        self.fit_params.update_values()
        
        # check for any fixed, constrained values
        for name in self.fit_params:
            p = self.fit_params[name]
            
            # if no dependencies are free, set vary = False, constrained = False
            if p.constrained:
                if not any(self.fit_params[dep].vary for dep in p.deps):
                    p.vary = False
                    p.constrained = False
            
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
            self.fit_params.update_values(**dict(zip(self.free_names, theta)))
        except Exception as e:
            import traceback
            msg = "exception while trying to update free parameters:\n"
            msg += "   current parameters:\n%s\n" %str(self.fit_params)
            msg += "   traceback:\n%s" %(traceback.format_exc())
            raise RuntimeError(msg)
        try:
            self.update_model()
        except Exception as e:
            import traceback
            msg = "exception while trying to update the theoretical model:\n"
            msg += "   current parameters:\n%s\n" %str(self.fit_params)
            msg += "   traceback:\n%s" %(traceback.format_exc())
            raise RuntimeError(msg)
        
        return True
        
    def model_callable(self, data):
        """
        Return the correct model function based on the `PowerData` 
        """
        import functools
        
        # if `data.transfer` is there, use it to evaluate power, accounting
        # for binning effects
        if data.transfer is not None: 
            return functools.partial(self.model.from_transfer, data.transfer, flatten=True)
        
        # computing P(k,mu) with no binning effects
        if data.mode == 'pkmu':
            k = data.combined_k
            mu = data.combined_mu
            return functools.partial(self.model.Pgal, k, mu)

        # computing P(k, ell) with no binning effects (i.e., integrating over P(k,mu))
        elif mode == 'poles':
            k = data.combined_k
            ell = data.combined_ell
            return functools.partial(self.model.Pgal_poles, k, ell)
        
        # all is lost...
        # something has gone horribly wrong...
        raise NotImplementedError("failure trying to get model callable... all is lost")
    
    def update_model(self):
        """
        Update the theoretical model with the values currently in 
        `GalaxyPowerTheory.fit_params`
        """
        self.model.update(**self.fit_params.to_dict())
    
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
