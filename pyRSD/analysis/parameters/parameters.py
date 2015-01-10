from . import tools
from ... import numpy as np

from collections import OrderedDict, defaultdict
import functools
import itertools
import string

#-------------------------------------------------------------------------------
class ParameterSet(OrderedDict):
    """
    A dictionary of all the Parameters required to specify a fit model.
    All keys must be strings and all values must be Parameters.
    """
    def __init__(self, *args, **kwargs):
        super(ParameterSet, self).__init__(self)
        
        # keep track of constraints
        self.constraints = defaultdict(dict)
        
        if (len(args) == 1 and isinstance(args[0], basestring)):
            self.load(args[0])
        else:
            self.update(*args, **kwargs)
        
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        # first get the parameters
        toret = "Parameters\n" + "_"*10 + "\n"
        toret += "\n".join(map(str, self.values()))
        
        # now get any constraints
        if sum(len(self.constraints[k]) for k in ['value', 'min', 'max']):
            toret += "\n\nConstraints\n" + "_"*10 + "\n"
            for constraint_type in ['value', 'min', 'max']:
                for name, text in self.constraints[constraint_type].iteritems():
                    toret += "%s\n" %(self._get_text_constraint(name, text, constraint_type))
        
        return toret
        
    #--------------------------------------------------------------------------
    def __call__(self, key):
        if not isinstance(key, int): 
            raise KeyError("Key must be an integer")
        if key >= len(self): 
            raise KeyError("ParameterSet only has size %d" %len(self))
        return self[self.keys()[key]]

    #---------------------------------------------------------------------------
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            if all(k in Parameter._valid_keys for k in value.keys()):
                value['name'] = key
                value = Parameter(**value)
        if value is not None and not isinstance(value, Parameter):
            value = Parameter(value=value, name=key)
        if key not in self:
            OrderedDict.__setitem__(self, key, value)
        else:
            self.update_param(**value)

    #---------------------------------------------------------------------------
    def __repr__(self):
        return "<ParameterSet (size: %d)>" %len(self)
        
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
    #end _verify_line
    
    #---------------------------------------------------------------------------
    def _get_text_constraint(self, name, text, constraint_type):
        """
        Return the text of a constraint
        """
        if constraint_type == 'value':
            op = "="
        elif constraint_type == 'min':
            op = "<" if self[name].exclusive_min else "<="
        elif constraint_type == 'max':
            op = ">" if self[name].exclusive_max else ">="
        
        return " ".join([name, op, text])
    #end _get_text_constraint
    
    #---------------------------------------------------------------------------
    def _parse_bounds(self, line, lineno, op):
        """
        Parse a bounds statement from file
        """
        lindex = line.index(op)
        rindex = line.rindex(op)
        if lindex == rindex:
            # first determine the operator
            split_on = op
            exclusive = True
            if line[lindex+1] == "=": 
                split_on = op + "="
                exclusive = False
                
            # do the split and verify
            line = line.split(split_on)
            line = [x.strip() for x in line]
            if not self._verify_line(line, 2, lineno): return
            
            # add the right limit
            if op == "<":
                self.add_upper_limit(line[0], line[-1], exclusive=exclusive)
            elif op == ">":
                self.add_lower_limit(line[0], line[-1], exclusive=exclusive)
        else:
            lexclusive = rexclusive = True
            lsplit_on = op
            if line[lindex+1] == "=": 
                lsplit_on = op + "="
                lexclusive = False
            rsplit_on = op
            if line[rindex+1] == "=": 
                rsplit_on = op + "="
                rexclusive = False
            lsplit = line.split(lsplit_on, 1)
            lsplit = [x.strip() for x in lsplit]
            if not self._verify_line(lsplit, 2, lineno): return
            
            rsplit = lsplit[-1].split(rsplit_on, 1)
            rsplit = [x.strip() for x in rsplit]
            if not self._verify_line(rsplit, 2, lineno): return
            
            key = rsplit[0]
            if op == "<":
                self.add_lower_limit(key, lsplit[0], exclusive=lexclusive)
                self.add_upper_limit(key, rsplit[-1], exclusive=rexclusive)
            elif op == ">":
                self.add_upper_limit(key, lsplit[0], exclusive=lexclusive)
                self.add_lower_limit(key, rsplit[-1], exclusive=rexclusive)
    #end _parse_bounds
    
    #---------------------------------------------------------------------------
    def load(self, filename, clear_current=False):
        """
        Fill with the parameters specified in the filename. If 
        `clear_current` is `True`, first empty current parameter settings.        
        """
        # get the correct path
        filename = tools.find_file(filename)

        D = {} 
        special_lines = {}
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
            
            # grab the special lines for later
            if "==" in line or any(op in line for op in ['<', '>']):
                special_lines[linecount] = line
                continue
                
            line = line.split('=')
            line = [x.strip() for x in line]

            if not self._verify_line(line, 2, linecount):
                continue

            # check for variables in the value
            if '$' in line[1]: line[1] = tools.replace_vars(line[1], D)

            # check for any functions calls in the line
            modules = tools.import_function_modules(line[1])

            # now save to the dict, eval'ing the line
            D[line[0].strip()] = eval(line[1].strip(), globals().update(modules), D)
                    
        if clear_current:
            self.clear()
        self.update(D)
        
        # now parse the special lines
        for lineno, line in special_lines.iteritems():
            # add the constraint
            if "==" in line:
                line = line.split('==')
                line = [x.strip() for x in line]

                if not self._verify_line(line, 2, lineno):
                    continue
                
                self.add_constraint(line[0].strip(), line[1].strip())
            else:
                # do bounds
                if "<" in line:
                    self._parse_bounds(line, lineno, "<")
                elif ">" in line:
                    self._parse_bounds(line, lineno, ">")
                else:
                    raise ValueError("Error parsing constraint/bound on line %d" %lineno)
                    
    #end load

    #---------------------------------------------------------------------------
    def _set_function_as_attribute(self, name, text, att_name):
        """
        Set the attribute provided to a constraining function
        
        Returns
        -------
        sucesss : bool
            `True` is returned if the text contains references to `Parameters`
            in the `ParameterSet`, or `False` otherwise
        """
        # the parameter names to be replaced
        param_names = set(tools.text_between_chars(text))
            
        if name in param_names and att_name == 'value':
            raise ValueError("Cannot reference own parameter name when setting value constraint")

        # any modules needing importing
        modules = tools.import_function_modules(text)
    
        # bind the parameters
        args = (self, text, param_names, modules)
        f = functools.partial(tools.constraining_function, *args)
        
        if name not in self:
            self[name] = Parameter(name=name)
        
        if len(param_names) == 0:
            setattr(self[name], att_name, f())
        else:
            setattr(self[name], att_name, f)
        
        return param_names, modules
    #end _set_function_as_attribute
    
    #---------------------------------------------------------------------------
    def _set_bounds_from_constraint(self, name, constraint, param_names, modules):
        """
        Set the bounds of a parameter `name` based on the constraint 
        """
        extrema = ['min', 'max']
        combos = list(itertools.combinations_with_replacement(extrema, len(param_names)))

        values = []
        for combo in combos:
            formatted_constraint = constraint.format(**{k:self[k][att] for k, att in zip(param_names, combo)})
            value = eval(formatted_constraint, globals().update(modules))
            values.append(value)
        
        min_combo = combos[values.index(min(values))]
        max_combo = combos[values.index(max(values))]    
        
        args = (self, constraint, param_names, modules)
        fmin = functools.partial(tools.constraining_function, *args, props=list(min_combo))
        fmax = functools.partial(tools.constraining_function, *args, props=list(max_combo))
        
        setattr(self[name], 'min', fmin)
        setattr(self[name], 'max', fmax)
    #end _set_bounds_from_constraint
        
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
    #end update_param
          
    #---------------------------------------------------------------------------
    def add_constraint(self, name, constraint):
        """
        Add an explicit constraint to the parameter set. If `name` is not 
        defined in the `ParameterSelf`, add the `Parameter` to the set
        
        No constraint is added if there are no parameter dependencies -- the 
        value is just set to the constant
        """            
        param_depends, modules = self._set_function_as_attribute(name, constraint, 'value')
        if len(param_depends):
            self.constraints['value'][name] = constraint
            
            # also need to set min/max based on parameter dependencies
            self._set_bounds_from_constraint(name, constraint, param_depends, modules)
        else:
            # delete constraint text stored
            self.constraints['value'].pop(name, None)
            
            self._try_remove_limits(name)
    #end add_constraint
    
    #---------------------------------------------------------------------------
    def _try_remove_limits(self, name):
        """
        Try to remove any bounds that are callable
        """
        # remove any constraint-based bounds
        try:
            self.remove_upper_limt(name)
        except:
            pass
        try:
            self.remove_lower_limt(name)
        except:
            pass
            
    #---------------------------------------------------------------------------
    def remove_constraint(self, name, val=None):
        """
        Remove a constraint from the `Parameter` specified by `name`
        """
        if name not in self:
            raise ValueError("Parameter `%s` not in this ParameterSet" %name) 
        if not self[name].constrained:
            raise ValueError("Parameter `%s` is not constrained" %name)
        
        self[name].remove_constraint(val)
        self.constraints['value'].pop(name)
        self._try_remove_limits(name)
        
    #end remove_constraint
    
    #---------------------------------------------------------------------------
    def add_upper_limit(self, name, upper_lim, exclusive=True):
        """
        Add an explicit upper limit to the parameter set. If `name` is not 
        defined in the `ParameterSelf`, add the `Parameter` to the set
        
        No constraint is added if there are no parameter dependencies -- the 
        limit is just set to the constant
        """
        if isinstance(upper_lim, basestring):            
            param_depends, _ = self._set_function_as_attribute(name, upper_lim, 'max')
            if len(param_depends):
                self.constraints['max'][name] = upper_lim
            else:
                self.constraints['max'].pop(name, None)
        else:
            if name not in self:
                self[name] = Parameter(name=name)
            self[name].max = upper_lim
            self.constraints['max'].pop(name, None)
            
        self[name].exclusive_max = exclusive
     #end add_upper_limit
    
    #---------------------------------------------------------------------------
    def add_lower_limit(self, name, lower_lim, exclusive=True):
        """
        Add an explicit lower limit to the parameter set. If `name` is not 
        defined in the `ParameterSelf`, add the `Parameter` to the set
        
        No constraint is added if there are no parameter dependencies -- the 
        limit is just set to the constant
        """            
        if isinstance(lower_lim, basestring):            
            param_depends, _ = self._set_function_as_attribute(name, lower_lim, 'min')
            if len(param_depends):
                self.constraints['min'][name] = lower_lim
            else:
                self.constraints['min'].pop(name, None)
        else:
            if name not in self:
                self[name] = Parameter(name=name)
            self[name].min = lower_lim
            self.constraints['min'].pop(name, None)
            
        self[name].exclusive_min = exclusive
    #end add_lower_limit
     
    #-------------------------------------------------------------------------------
    def remove_upper_limit(self, name, val=None):
        """
        Remove the upper limit from the `Parameter` specified by `name`
        """
        if name not in self:
            raise ValueError("Parameter `%s` not in this ParameterSet" %name) 
        if self[name].max is None:
            raise ValueError("Parameter `%s` is does not have an upper limit" %name)
        
        self[name].remove_upper_limit(val)
    #end remove_upper_limit
    
    #---------------------------------------------------------------------------
    def remove_lower_limit(self, name, val=None):
        """
        Remove the lower limit from the `Parameter` specified by `name`
        """
        if name not in self:
            raise ValueError("Parameter `%s` not in this ParameterSet" %name) 
        if self[name].min is None:
            raise ValueError("Parameter `%s` is does not have an lower limit" %name)
        
        self[name].remove_lower_limit(val)
    #end remove_lower_limit
    
    #---------------------------------------------------------------------------
    @property
    def free_parameter_names(self):
        """
        Return the free parameter names. `Free` means that `vary = True` and 
        `constrained = False`
        """
        return [k for k in self if self[k].vary and not self[k].constrained]
    
    #---------------------------------------------------------------------------
    @property
    def free_parameter_values(self):
        """
        Return the free parameter values. `Free` means that `vary = True` and 
        `constrained = False`
        """
        return [self[k].value for k in self if self[k].vary and not self[k].constrained]
    
    #---------------------------------------------------------------------------
    @property
    def free_parameters(self):
        """
        Return the free `Parameter` objects. `Free` means that `vary = True` and 
        `constrained = False`
        """
        return [self[k] for k in self if self[k].vary and not self[k].constrained]
    
    #---------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert the parameter set to a dictionary using (`name`, `value`)
        as the (key, value) pairs
        """
        return {k : self[k].value for k in self}
#endclass ParameterSet

#-------------------------------------------------------------------------------
class Parameter(object):
    """
    A class to represent a (possibly bounded) generic parameter. The prior
    is taken to be a uniform distribution between the min/max allowed
    parameter values
    """
    _valid_keys = ['name', 'value', 'fiducial_value', 'vary', 'min', 'max', 
                   'exclusive_min', 'exclusive_max', 'derived', 'description']
                   
    def __init__(self, name=None, **props):
                
        # no name is given, this is bad...
        if name is None:
            raise ValueError('`Parameter` instance needs at least a `name` as an argument')
        
        # store the name
        props['name'] = name

        # setup default properties
        props.setdefault('description', 'No description available')
        props.setdefault('derived', False)
        props.setdefault('value', None)
        props.setdefault('vary', False)
        props.setdefault('exclusive_min', False)
        props.setdefault('exclusive_max', False)
        props.setdefault('fiducial_value', None)
        
        # remember initial settings
        self._initial = props.copy()
        
        # attach all keys to the class instance
        self.reset()    
        
        
    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        return getattr(self, key)    
    
    #---------------------------------------------------------------------------
    def keys(self):
        return [k for k in self._valid_keys]
        
    #---------------------------------------------------------------------------
    def __call__(self):
        """
        Returns `self.value`
        """
        return self.value
        
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
        s = []
        s.append("%-15s" %("'" + self.name + "'"))
        sval = tools.format(self.value)
        sval = "value=%-15.5g" %sval
        
        fid_tag = ", fiducial" if self.value == self.fiducial_value else ""
        if self.constrained:
            t = " (constrained)"
        elif self.derived:
            t = " (derived)"
        elif not self.vary:
            t = " (fixed%s)" %fid_tag
        else:
            t = " (free%s)" %fid_tag
        sval += "%-20s" %t

        s.append(sval)
        s.append("bounds=[%10.5g:%10.5g]" % (tools.format(self.min), tools.format(self.max)))
        return "<Parameter %s>" % ' '.join(s)
    
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        return self.__repr__()    
    
    #---------------------------------------------------------------------------
    def reset(self):
        """
        Reset the parameter values to its initial values.
        """
        for key in self._valid_keys:
            if key in self._initial:
                setattr(self, key, self._initial[key])
            elif hasattr(self, key):
                setattr(self, key, None)

    #---------------------------------------------------------------------------
    def remember(self):
        """
        Set the current properties as initial value.
        """
        for key in self._valid_keys:
            if hasattr(self, key):
                self._initial[key] = getattr(self, key)
        
    #---------------------------------------------------------------------------
    def clear(self):
        """
        Strip this instance from its properties
        """
        for key in self._initial:
            delattr(self, key)
    
    #---------------------------------------------------------------------------
    @property
    def min(self):
        """
        The minimum allowed value for this parameter (inclusive)
        """
        try:
            if callable(self._min):
                return self._min()
            else:
                return self._min
        except AttributeError:
            return None
    
    @min.setter
    def min(self, val):
        if val == -np.inf: 
            val = None
        self._min = val
        
    def remove_lower_limit(self, val=None):
        """
        Remove the lower limit on this parameter. If `val = None`, this will set 
        `self._min` to the current value returned by the constraining 
        function, otherwise it will set it to `val`
        """
        if self.min is not None:
            self.min = self.min if val is None else val
    
    #---------------------------------------------------------------------------
    @property
    def max(self):
        """
        The maximum allowed value for this parameter (inclusive)
        """
        try:
            if callable(self._max):
                return self._max()
            else:
                return self._max
        except AttributeError:
            return None
    
    @max.setter
    def max(self, val):
        if val == np.inf: 
            val = None
        self._max = val
    
    def remove_upper_limit(self, val=None):
        """
        Remove the upper limit on this parameter. If `val = None`, this will set 
        `self._max` to the current value returned by the constraining 
        function, otherwise it will set it to `val`
        """
        if self.max is not None:
            self.max = self.max if val is None else val
    
    #---------------------------------------------------------------------------
    @property
    def within_bounds(self):
        """
        Returns `True` if the current value is within the bounds
        """
        if not self.bounded:
            return True
            
        if self.exclusive_min:
            min_cond = self.value > self.min
        else:
            min_cond = self.value >= self.min
            
        if self.exclusive_max:
            max_cond = self.value < self.max
        else:
            max_cond = self.value <= self.max
        return max_cond and min_cond
              
    #---------------------------------------------------------------------------
    @property
    def bounded(self):
        """
        Parameter is bounded if (min, max) != (None, None)
        """
        return self.limits != (None, None)
    
    @property
    def limits(self):
        """
        Convenience property to return the limits
        """
        return self.min, self.max
    
    #---------------------------------------------------------------------------
    @property
    def value(self):
        """
        Return the value for this parameter, which could optionally be a
        the value returned by a constraining function. If no value has been 
        set, this returns the fiducial value by default
        """
        # if the parameter is derived, call the deriving function
        if self.derived:
            return self.deriving_function()
            
        try:
            if callable(self._value):
                return self._value()
            else:
                if self._value is None:
                    raise AttributeError()
                return self._value
        except AttributeError:
            if hasattr(self, 'fiducial_value') and self.fiducial_value is not None:
                return self.fiducial_value
            else:
                raise AttributeError("No value has been defined for parameter %s" %name)
    
    @value.setter
    def value(self, val):
        self._value = val    
    
    #---------------------------------------------------------------------------
    @property
    def constrained(self):
        """
        Parameter is constrained if the `value` property is evaluating a 
        function
        """
        return callable(self._value)
        
    def remove_constraint(self, val=None):
        """
        Remove the constraint on this parameter. If `val = None`, this will set 
        `self._value` to the current value returned by the constraining 
        function, otherwise it will set it to `val`
        """
        if self.constrained:
            self.value = self.value if val is None else val
    
    #---------------------------------------------------------------------------
    @property
    def lnprior(self):
        """
        Return log of the prior, which is a uniform prior between `self.min`
        and `self.max`
        """
        within_bounds = self.within_bounds
        return 0 if within_bounds else -np.inf
    
    #---------------------------------------------------------------------------
    def get_value_from_prior(self, size=1):
        """
        Get random values from the prior, of size `size`
        """
        if not self.bounded:
            raise ValueError("Cannot draw value from unbounded uniform prior")
        return np.random.uniform(size=size, low=self.min, high=self.max)
    
    #---------------------------------------------------------------------------
    def set_value_from_prior(self):
        """
        Set a random value from the prior.
        """
        try:
            value = self.get_value_from_prior()[0]
        except ValueError:
            return None
        self.value = value
        
    #---------------------------------------------------------------------------
    def deriving_function(self):
        """
        Function that computes the value for the parameter, likely derived
        from other `Parameters` in a `ParameterSet`
        """
        raise NotImplementedError("Parameter %s cannot be derived; no function provided" %self.name)
    
    #---------------------------------------------------------------------------