"""
    parameterset.py
    pyRSD.rsdfit.parameters

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : A class to store a set of Parameters
"""

import collections
import string
import copy
import copyreg

from . import tools, Parameter
from ...extern import lmfit
from ... import os, numpy as np

class PickeableClass(type):
    def __init__(cls, name, bases, attrs):
        copyreg.pickle(cls, _pickle, _unpickle)

def _pickle(params):    
    items = [[k, params[k]] for k in params]
    inst_dict = vars(params).copy()
    for k in vars(collections.OrderedDict()):
        inst_dict.pop(k, None)
    inst_dict.pop('_asteval')
    return _unpickle, (params.__class__, items, inst_dict, )

def _unpickle(cls, items, meta):
    toret = cls()
    for k in meta: setattr(toret, k, meta[k])
    toret.update(items)
    for k, v in toret._registered_functions.iteritems():
        toret.register_function(k, v)
    toret.prepare_params()
    try:
        toret.update_values()
    except:
         pass
        
    return toret

class ParameterSet(lmfit.Parameters, metaclass=PickeableClass):
    """
    A subclass of `lmfit.Parameters` that adds the ability to update values
    based on constraints in place
    """    
    def __init__(self, *args, **kwargs):
        super(ParameterSet, self).__init__(*args, **kwargs)

        self._asteval = lmfit.asteval.Interpreter()
        self._namefinder = lmfit.astutils.NameFinder()
        self._prepared = False
        self._registered_functions = {}
        self.tag = None
        
    #---------------------------------------------------------------------------
    # builtin functions
    #---------------------------------------------------------------------------
    def __str__(self):
        # first get the parameters
        toret = "Parameters\n" + "_"*10 + "\n"
        toret += "\n".join(map(str, sorted(self.values(), key=lambda p: p.name.lower())))
        
        # now get any constraints
        toret += "\n\nConstraints\n" + "_"*10 + "\n"
        for name in self:
            if self[name].expr is not None:
                toret += "%s = %s\n" %(name, self[name].expr)
        return toret
        
    def __call__(self, key):
        """
        Return the value of the parameter, specified either by the integer
        value `key`, or the name of the parameter
        
        Parameters
        ----------
        key : str or int
            either a string specifying the name of the parameter, or the
            integer index of the parameter in the collection 
        """
        if not isinstance(key, (int, str)): 
            raise KeyError("key must either be an integer or a str")
        
        if isinstance(key, str):
            if key not in self:
                raise ValueError("no parameter with name `%s` in ParameterSet" %key)
            return self[key]()
        else:
            if key >= len(self): 
                raise KeyError("ParameterSet only has size %d" %len(self))
            key = self.keys()[key]
            return self[key]()

    def __repr__(self):
        """
        Builtin representation function
        """
        return "<ParameterSet (size: %d)>" %len(self)
        
    def copy(self):
        """
        Return a copy
        """
        return copy.deepcopy(self)
        
    @classmethod
    def from_file(cls, filename, tags=[]):
        """
        Read a file and return a `collections.defaultdict` with the keys given
        by `tags` and the values given by a `lmfit.Parameters` object
    
        Parameters
        ----------
        filename : str
            the name of the file to read parameters from
        tags : list, optional
            list of any parameter tags to specifically seach for
        """
        if isinstance(tags, str):
            tags = [tags]
        if len(tags) > 1:
            toret = collections.defaultdict(cls)
        else:
            toret = cls()
            if len(tags) == 1: toret.tag = tags[0]

        # check the path
        if not os.path.exists(filename):
            raise IOError("no file found at path %s" %filename)
        
        orig = {}   
        D = {} 
        old = ''
        # loop over each line
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

            if not tools.verify_line(line, 2, linecount):
                continue
        
            # get the key
            key = line[0].strip()
            
            # check for variables in the value
            save_orig = None
            if '$' in line[1]: 
                save_orig = eval(line[1].strip())
                line[1] = tools.replace_vars(line[1], D)
            
            # check for any functions calls in the line
            modules = tools.import_function_modules(line[1])

            # now save to the dict, eval'ing the line
            D[key] = eval(line[1].strip(), globals().update(modules), D)
            if save_orig is not None:
                orig[key] = save_orig
                
        # now make the output
        valid_kwargs = ['name', 'value', 'vary', 'min', 'max', 'expr']
        for k, v in D.iteritems():
            if isinstance(v, dict) and all(kw in Parameter.valid_keys for kw in v):
                v = Parameter(**v)
            else:
                v = Parameter(value=v)
            v.name = k
            
            if k in orig:
                v.output_value = orig[k]
        
            # check for a possible key tag
            if len(tags):
                matched = False
                for tag in tags:
                    split_keys = k.rsplit(tag+'.')
                    if len(split_keys) != 2:
                        continue
                    else:
                        key = split_keys[-1].strip()
                        v.name = key
                        matched = True
                        break
                if matched:
                    if len(tags) > 1: 
                        toret[tag][key] = v
                    else:
                        toret[key] = v
            else:  
                try:
                    k = k.replace('.', '_')
                    toret[k] = v
                except Exception as e:
                    pass
        
        if len(tags) > 1:
            for tag in tags:
                toret[tag].tag = tag
                
        return toret
                        
    def to_file(self, filename, mode='w', header_name=None, footer=False, as_dict=True):
        """
        Output the `ParameterSet` to a file, using the mode specified. 
        Optionally, add a header and/or footer to make it look nice.
        
        If `as_dict = True`, output the parameter as a dictionary, otherwise
        just output the value
        
        Parameters
        ----------
        """
        if hasattr(filename, 'write'):
            f = filename
            close = False
        else:
            f = open(filename, mode=mode)
            close = True
        
        if header_name is not None:
            header = "#{x}\n# {hdr}\n#{x}\n".format(hdr=header_name, x="-"*79)
            f.write(header)
    
        output = []
        for name, par in sorted(self.items()):
            key = name if self.tag is None else "{}.{}".format(self.tag, name)
            if as_dict:
                par_dict = par.to_dict(output=True)
                if len(par_dict):
                    output.append("{} = {}".format(key, par_dict))
            else:
                output.append("{} = {}".format(key, repr(par(output=True))))
        
        f.write("%s\n" %("\n".join(output)))
        if footer: f.write("#{}\n\n".format("-"*79))
        
        if close: f.close()
        
    #---------------------------------------------------------------------------
    # functions to handle param constraints
    #---------------------------------------------------------------------------
    def register_function(self, name, function):
        """
        Register a function in the ``symtable`` of the ``asteval`` attribute
        """
        if not hasattr(self, '_asteval'):
            self._asteval = lmfit.asteval.Interpreter()
        self._asteval.symtable[name] = function
        self._registered_functions[name] = function
        
    def prepare_params(self):
        """
        Prepare the parameters by parsing the dependencies. We initialize the 
        ``ast`` classes in order to evaluate any constrained parameters
        """
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
        self._prepared = True
        
    def update_fiducial(self):
        """
        Update the fiducial values of the constrained parameters
        """
        values = self.valuesdict()

        for name in self:
            if self[name].expr is None:
                fid = self[name].fiducial
                self[name].value = fid if fid is not None else np.inf
            
        self.update_values()
        for name in self:
            if self[name].expr is not None:
                val = self[name].value
                if val is not None and not np.isfinite(val): val = None 
                self[name].fiducial = val
            
        for k,v in values.items():
            if self[k].expr is None:
                self[k].value = v
                
    def update_values(self):
        """
        Update the values of all parameters, checking that dependencies are
        evaluated as needed.
        """    
        if not self._prepared:
            raise RuntimeError("cannot call ``update`` before calling ``prepare_params``")
        self._updated = dict([(name, False) for name in self])
        for name in self:
            self._update_parameter(name)
        
    def _update_parameter(self, name):
        """
        Internal function to recursively update parameter called ``name``, 
        accounting for any constraints on this parameter.
        """
        if self._updated[name]:
            return
            
        par = self[name]
        if getattr(par, 'expr', None) is not None:
            if getattr(par, 'ast', None) is None:
                par.ast = self._asteval.parse(par.expr)
            if par.deps is not None:
                for dep in par.deps:
                    self._update_parameter(dep)
            try:
                par.value = self._asteval.run(par.ast, with_raise=True)  
            except:
                msg = "error evaluating '%s' for parameter '%s'" %(par.expr, name)
                raise ValueError(msg)

        self._asteval.symtable[name] = par.value
        self._updated[name] = True
    
    #---------------------------------------------------------------------------
    # convenient functions/attributes
    #---------------------------------------------------------------------------
    def add(self, name, **kwargs):
        """
        Add a `Parameter` with `name` to the ParameterSet with the specified 
        `value`.
        """
        if name in self:
            self.update_param(name, **kwargs)
        else:
            self[name] = Parameter(name=name, **kwargs)
        
    def to_dict(self):
        """
        Convert the parameter set to a dictionary using (`name`, `value`)
        as the (key, value) pairs
        """
        return {k : self[k]() for k in self}
        
    def get(self, name, default=None):
        """
        Mirrors the `dict.get()` method behavior, but returns the parameter values
        """
        return self[name]() if name in self else default
        
    def update_param(self, *name, **kwargs):
        """
        Update the `Parameter` specified by `name` with the keyword arguments
        provided
        """
        if len(name) == 0 and 'name' not in kwargs:
            raise ValueError("please specify the name of the parameter to update")
        
        if len(name) == 1:
            name = name[0]
            if 'name' in kwargs: kwargs.pop('name')
        else:
            name = kwargs.pop('name')
        
        if name not in self:
            raise ValueError("Parameter `%s` not in this ParameterSet" %name)
        for k, v in kwargs.iteritems():
            setattr(self[name], k, v)
              
    @property
    def free_dtype(self):
        """
        The list of 'free' data types for the structured array
        """
        return [self[k].dtype for k in self.free_names]
    
    @property
    def free_names(self):
        """
        Return the free parameter names. `Free` means that 
        `Parameter.vary = True` and `Parameter.constrained = False`
        """
        return [k for k in self if self[k].vary and not self[k].constrained]
    
    @property
    def free_values(self):
        """
        Return the free parameter values. `Free` means that 
        `Parameter.vary = True` and `Parameter.constrained = False`
        """
        return [self[k]() for k in self.free_names]
    
    @property
    def free(self):
        """
        Return a list of the free `Parameter` objects. `Free` means that 
        `Parameter.vary = True` and `Parameter.constrained = False`
        """
        return [self[k] for k in self.free_names]
    
    @property
    def constrained_names(self):
        """
        Return the constrained parameter names. `Constrained` means that 
        `Parameter.constrained = True`
        """
        return [k for k in self if self[k].constrained]
    
    @property
    def constrained_dtype(self):
        """
        The list of 'constrained' data types for the structured array
        """
        return [self[k].dtype for k in self.constrained_names]
        
    @property
    def constrained_values(self):
        """
        Return the constrained parameter values.
        """
        data = tuple(self[k]() for k in self.constrained_names)
        return np.array(data, dtype=self.constrained_dtype)
    
    @property
    def constrained(self):
        """
        Return a list of the constrained `Parameter` objects.
        """
        return [self[k] for k in self.constrained_names]

    
        
        
        
        
        
        