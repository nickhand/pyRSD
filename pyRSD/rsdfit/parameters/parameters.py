from . import tools, distributions as dists
from ... import numpy as np
from ...rsd._cache import Cache, parameter, cached_property

from collections import OrderedDict, defaultdict
import itertools
import string
import copy
import lmfit

#-------------------------------------------------------------------------------
class ParameterSet(OrderedDict):
    """
    A subclass of `collectiosn.OrderedDict` that stores the `Parameter` objects 
    required to specify a fit model. All keys must be strings and all the
    values must be `Parameters`.
    
    Notes
    -----
    In order to accurately handle constrained parameters, parameter values
    should be set through the `set(name, value)` function, which will 
    automatically update any constrained parameters that depend on the 
    parameter `name` 
    """
    def __init__(self, filename, tag=None, update_on_init=True):
        """
        Initialize the `ParameterSet`
        
        Parameters
        ----------
        filename : str
            The name of the file to read the parameters from
        tag : str, optional
            If not `None`, only read the parameters with this prefix in their
            key name
        update_on_init : bool, optional
            Update any constraints on initialization
        """
        # initialize the base class
        super(ParameterSet, self).__init__()
        self.tag = tag
        
        # load the parameters
        self.load(filename)
        
        # initialize the constraint readers and update constraints
        self._asteval = lmfit.asteval.Interpreter()
        self._namefinder = lmfit.astutils.NameFinder()
        self._update_dependencies()
        if update_on_init:
            self.update_constraints()
        

    #---------------------------------------------------------------------------
    def to_file(self, filename, mode='w', header_name=None, footer=False, as_dict=True):
        """
        Output the `ParameterSet` to a file, using the mode specified. 
        Optionally, add a header and/or footer to make it look nice.
        
        If `as_dict = True`, output the parameter as a dictionary, otherwise
        just output the value
        """
        # open the file
        f = open(filename, mode=mode)
        
        if header_name is not None:
            header = "#{x}\n# {hdr}\n#{x}\n".format(hdr=header_name, x="-"*79)
            f.write(header)
        
        output = []
        for name, par in self.items():
            key = name if self.tag is None else "{}.{}".format(self.tag, name)
            
            if as_dict:
                par_dict = par.to_dict()
                if len(par_dict):
                    output.append("{} = {}".format(key, par_dict))
            else:
                output.append("{} = {}".format(key, repr(par())))
            
        f.write("%s\n" %("\n".join(output)))
        
        if footer:
            f.write("#{}\n\n".format("-"*79))
        f.close()
        
    #---------------------------------------------------------------------------
    def _update_dependencies(self):
        """
        Prepare and save the parameters. The important step here is initializing
        the `asteval` and `ast` attributes so any parameters with `expr` can
        be evaluated
        """
        # for each parameter, this will store the other params that depend
        # on it
        self._dependencies = defaultdict(set)
        
        # check for any parameters that have `expr` defined
        for name, par in self.items():
            if par.expr is not None:
                par.ast = self._asteval.parse(par.expr)
                par.vary = False
                par.deps = []
                self._namefinder.names = []
                self._namefinder.generic_visit(par.ast)
                for symname in self._namefinder.names:
                    if (symname in self and symname not in par.deps):
                        par.deps.append(symname)
                        self._dependencies[symname].add(name)
        
    #---------------------------------------------------------------------------
    def add_constraint(self, name, expr, update_constraints=True):
        """
        Update the parameter `name` with the constraining expression `expr`, and
        update then update the constraints
        """
        if name not in self:
            raise ValueError("Cannot add constraint for parameter `%s`; not in ParameterSet" %name)
        
        # update the dependencies
        self[name].expr = expr
        self._update_dependencies()
        
        if update_constraints:
            self.update_constraints()
        
    #---------------------------------------------------------------------------
    def update_constraints(self):
        """
        Update all constrained parameters, checking that dependencies are
        evaluated as needed.
        """    
        self._updated = dict([(name, False) for name in self])
        for name in self:
            self._update_param_value(name)
            
    #---------------------------------------------------------------------------
    def _update_param_value(self, name):
        """
        Update parameter value for a constrained parameter, which is a parameter
        that has `expr` defined. This first updates (recursively) all parameters 
        on which the parameter depends (using the 'deps' field).
        """
        if self._updated[name]:
            return
        
        par = self[name]
        if getattr(par, 'expr', None) is not None:
            if getattr(par, 'ast', None) is None:
                par.ast = self._asteval.parse(par.expr)
            if par.deps is not None:
                for dep in par.deps:
                    self._update_param_value(dep)
            par.value = self._asteval.run(par.ast)
        self._asteval.symtable[name] = par.value
        self._updated[name] = True
        
    #---------------------------------------------------------------------------
    def __getstate__(self):
        """
        Make the class pickeable by deleting the `asteval` class first
        """
        # delete the non-pickeable stuff
        if hasattr(self, '_asteval'): delattr(self, '_asteval')
        d =  self.__dict__.copy()
        return d

    #---------------------------------------------------------------------------
    def __setstate__(self, d):
        """
        Restore the class after pickling, and update the constraints
        """
        self.__dict__ = d

        # restore the asteval and update constraints
        self._asteval = lmfit.asteval.Interpreter()
        self.update_constraints()

    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        # first get the parameters
        toret = "Parameters\n" + "_"*10 + "\n"
        toret += "\n".join(map(str, sorted(self.values(), key=lambda p: p.name.lower())))
        
        # now get any constraints
        toret += "\n\nConstraints\n" + "_"*10 + "\n"
        for name in self:
            if self[name].expr is not None:
                toret += "%s = %s\n" %(name, self[name].expr)
        
        return toret
        
    #--------------------------------------------------------------------------
    def __call__(self, key):
        """
        Return the value of the parameter, specified either by the integer
        value `key`, or the name of the parameter
        """
        if not isinstance(key, (int, basestring)): 
            raise KeyError("Key must either be an integer or a basestring")
        
        if isinstance(key, basestring):
            if key not in self:
                raise ValueError("No parameter with name `%s` in ParameterSet" %key)
            return self[key]()
        else:
            if key >= len(self): 
                raise KeyError("ParameterSet only has size %d" %len(self))
            key = self.keys()[key]
            return self[key]()

    #---------------------------------------------------------------------------
    def __setitem__(self, key, value):
        """
        Set the item in the dictionary, making sure all items are `Parameters`
        """
        if isinstance(value, dict):
            if all(k in Parameter._valid_keys for k in value.keys()):
                value['name'] = key
                value = Parameter(**value)
        if not isinstance(value, Parameter):
            value = Parameter(value=value, name=key)
        if key not in self or self[key] is None:
            OrderedDict.__setitem__(self, key, value)
        else:
            self.update_param(**value)

    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation function
        """
        return "<ParameterSet (size: %d)>" %len(self)
        
    #---------------------------------------------------------------------------
    def copy(self):
        """
        Return a copy
        """
        return copy.copy(self)
        
    #---------------------------------------------------------------------------
    def _verify_line(self, line, length, lineno):
        """
        Verify the line read makes sense
        """
        if (len(line) == 0 or line[0] == ''): 
            return False
        if len(line) == 1:
            raise ValueError("Error reading parameter %s on line %d" %(line[0], lineno))
        elif len(line) != length:
            raise ValueError("Cannot understand line %d" %lineno)
        return True
    
    #---------------------------------------------------------------------------
    def get(self, name, default=None):
        """
        Mirrors the `dict.get()` method behavior, but returns the parameter values
        """
        return self[name]() if name in self else default
        
    #---------------------------------------------------------------------------
    def _update_constrained_params(self, name):
        """
        Update the constrained params that depend on `name`, recursively
        """
        if name not in self._dependencies:
            return
            
        constrained_params = self._dependencies[name]
        for constrained_name in constrained_params:
            self._updated[constrained_name] = False
            self._update_param_value(constrained_name)
            self._update_constrained_params(constrained_name)
            
        
    #---------------------------------------------------------------------------
    def set(self, name, value, update_constraints=True):
        """
        Set the `Parameter` value with `name` to `value`. This will also update
        the values of any parameters are constrained and depend on the parameter
        given by `name`
        
        Notes
        -----
        The parameter given by `name` is assumed to already exist in the 
        `ParameterSet`
        """
        if name not in self:
            raise ValueError("Cannot set parameter `%s`; not in ParameterSet" %name)
        
        # set the value and update the symtable
        self[name].value = value
        self._asteval.symtable[name] = value
        
        # check for any constrained params that depend on this
        if update_constraints:
            self._updated = dict([(k, True) for k in self])
            self._update_constrained_params(name)
        
    #---------------------------------------------------------------------------
    def add(self, name, value):
        """
        Add a `Parameter` with `name` to the ParameterSet with the specified 
        `value`.
        """
        self[name] = Parameter(name=name, value=value)
            
    #---------------------------------------------------------------------------
    def load(self, filename, clear_current=False):
        """
        Fill with the parameters specified in the filename. If 
        `clear_current` is `True`, first empty current parameter settings.        
        """
        # get the correct path
        filename = tools.find_file(filename)
            
        D = {} 
        old = ''
        for linecount, line in enumerate(open(filename, 'r')):
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            s = line.split('#')
            line = s[0]
            s = line.split('\\')
            if len(s) > 1:
                old = string.join([old, s[0]])
                continue
            else:
                line = string.join([old, s[0]])
                old = ''
            for i in xrange(len(line)):
                if line[i] !=' ':
                    line = line[i:]
                    break
                
            line = line.split('=')
            line = [x.strip() for x in line]

            if not self._verify_line(line, 2, linecount):
                continue
            
            # check for a possible key tag
            if self.tag is not None:
                split_keys = line[0].strip().rsplit(self.tag+'.')
                if len(split_keys) != 2:
                    continue
                else:
                    key = split_keys[-1].strip()
            else:
                key = line[0].strip()
                
            # check for variables in the value
            if '$' in line[1]: line[1] = tools.replace_vars(line[1], D)

            # check for any functions calls in the line
            modules = tools.import_function_modules(line[1])

            # now save to the dict, eval'ing the line
            D[key] = eval(line[1].strip(), globals().update(modules), D)
                    
        if clear_current:
            self.clear()
        self.update(D)
                
    #---------------------------------------------------------------------------
    def update_param(self, *name, **kwargs):
        """
        Update the `Parameter` specified by `name` with the keyword arguments
        provided
        """
        if len(name) == 0 and 'name' not in kwargs:
            raise ValueError("Please specify the name of the parameter to update")
        
        if len(name) == 1:
            name = name[0]
            if 'name' in kwargs: kwargs.pop('name')
        else:
            name = kwargs.pop('name')
        
        if name not in self:
            raise ValueError("Parameter `%s` not in this ParameterSet" %name)
        for k, v in kwargs.iteritems():
            setattr(self[name], k, v)
              
    #---------------------------------------------------------------------------
    @property
    def free_parameter_names(self):
        """
        Return the free parameter names. `Free` means that 
        `Parameter.vary = True` and `Parameter.constrained = False`
        """
        try:
            return self._free_parameter_names
        except AttributeError:
            self._free_parameter_names = [k for k in self if self[k].vary and not self[k].constrained]
            return self._free_parameter_names
    
    #---------------------------------------------------------------------------
    @property
    def free_parameter_values(self):
        """
        Return the free parameter values. `Free` means that 
        `Parameter.vary = True` and `Parameter.constrained = False`
        """
        return np.array([self[k]() for k in self.free_parameter_names])
    
    #---------------------------------------------------------------------------
    @property
    def free_parameters(self):
        """
        Return the free `Parameter` objects. `Free` means that 
        `Parameter.vary = True` and `Parameter.constrained = False`
        """
        return [self[k] for k in self.free_parameter_names]
    
    #---------------------------------------------------------------------------
    @property
    def constrained_parameter_names(self):
        """
        Return the constrained parameter names. `Constrained` means that 
        `Parameter.constrained = True`
        """
        try:
            return self._constrained_parameter_names
        except AttributeError:
            self._constrained_parameter_names = [k for k in self if self[k].constrained]
            return self._constrained_parameter_names
    
    #---------------------------------------------------------------------------
    @property
    def constrained_parameter_values(self):
        """
        Return the free parameter values. `Free` means that `vary = True` and 
        `constrained = False`
        """
        return np.array([self[k].value for k in self.constrained_parameter_names])
    
    #---------------------------------------------------------------------------
    @property
    def constrained_parameters(self):
        """
        Return the free `Parameter` objects. `Free` means that `vary = True` and 
        `constrained = False`
        """
        return [self[k] for k in self.constrained_parameter_names]
    
    #---------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert the parameter set to a dictionary using (`name`, `value`)
        as the (key, value) pairs
        """
        return {k : self[k]() for k in self}
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class Parameter(Cache):
    """
    A class to represent a generic parameter, with a prior. Currently, the prior
    can either be a uniform or normal distribution.
    """
    _valid_keys = ['name', 'value', 'fiducial', 'vary', 'min', 'max', 'expr', 
                   'description', 'prior', 'lower', 'upper', 'mu', 'sigma']
                   
    #---------------------------------------------------------------------------
    def __init__(self, name=None, **kwargs):
        """
        Initialize the parameter
        
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
        # initialize the base class
        super(Parameter, self).__init__()
        
        # store the name
        if name is None:
            raise ValueError("Must supply a name for the `Parameter`")
        self.name = name

        # setup default parameters
        self.description = kwargs.get('description', 'No description available')
        self.vary        = kwargs.get('vary', False)
        self.fiducial    = kwargs.get('fiducial', None)
        self.value       = kwargs.get('value', self.fiducial)
        self.min         = kwargs.get('min', None)
        self.max         = kwargs.get('max', None)
        self.expr        = kwargs.get('expr', None)
        
        # set the prior-related parameters
        self.lower      = kwargs.get('lower', None)
        self.upper      = kwargs.get('upper', None)
        self.mu         = kwargs.get('mu', None)
        self.sigma      = kwargs.get('sigma', None)
        self.prior_name = kwargs.get('prior', None)
        
    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
    @parameter
    def name(self, val):
        """
        The parameter name
        """
        return val
    
    @parameter
    def description(self, val):
        """
        The parameter description
        """
        return val
    
    @parameter
    def vary(self, val):
        """
        Whether to vary the parameter
        """
        return val
    
    @parameter
    def fiducial(self, val):
        """
        The fiducial parameter value
        """
        return val
    
    @parameter
    def value(self, val):
        """
        The parameter value
        """
        return val
        
    @parameter
    def min(self, val):
        """
        The minimum allowed parameter value
        """
        return val    
        
    @parameter
    def max(self, val):
        """
        The maximum allowed parameter value
        """
        return val
        
    @parameter
    def expr(self, val):
        """
        Either a mathematical expression for evaluating the parameter or 
        a callable function that returns the parameter value
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
    
    #---------------------------------------------------------------------------
    # Cached properties
    #---------------------------------------------------------------------------
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
            raise NotImplementedError("Only `uniform` and `normal` priors currently implemented")
    
    @cached_property('prior')
    def has_prior(self):
        """
        Whether the parameter has a prior defined
        """
        return self.prior is not None
        
    @cached_property('value', 'bounded', 'min', 'max')
    def within_bounds(self):
        """
        Returns `True` if the current value is within the bounds
        """
        if not self.bounded:
            return True
            
        min_cond = True if self.min is None else self.value >= self.min
        max_cond = True if self.max is None else self.value <= self.max
        return min_cond and max_cond

    @cached_property('min', 'max')
    def bounded(self):
        """
        Parameter is bounded if either `min` or `max` is defined
        """
        return self.min is not None or self.max is not None
    
    @cached_property('expr')
    def constrained(self):
        """
        Parameter is constrained if the `Parameter.expr == None`
        """
        return self.expr is not None
            
    @cached_property('value', 'min', 'max', 'prior')
    def lnprior(self):
        """
        Return log of the prior, which is either a uniform or normal prior.
        If the current value is outside `Parameter.min` or `Parameter.max`, 
        return `numpy.inf`
        """
        lnprior = -np.inf
        if self.within_bounds:
            lnprior = 0
            if self.has_prior:
                lnprior = np.log(self.prior.pdf(domain=self.value)[1])
        return lnprior
    
    #---------------------------------------------------------------------------
    # Functions
    #---------------------------------------------------------------------------
    def __call__(self):
        """
        Returns `Parameter.value` when called
        """
        return self.value
        
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
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
    
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        return self.__repr__()    
            
    #---------------------------------------------------------------------------
    def keys(self):
        """
        Return the valid, allowed attribute names
        """
        return self._valid_keys
      
    #---------------------------------------------------------------------------
    def get_value_from_prior(self, size=1):
        """
        Get random values from the prior, of size `size`
        """
        if not self.has_prior:
            raise ValueError("Cannot draw from prior that does not exist")
            
        return self.prior.draw(size=size)
    
    #---------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert the `Parameter` object to a dictionary
        """
        toret = {}
        for k in self.keys():
            if k == 'description': continue
            if k == 'prior':
                val = getattr(self, 'prior_name')
            else:
                val = getattr(self, k)
                
            if val is not None:
                toret[k] = val
        return toret
#-------------------------------------------------------------------------------