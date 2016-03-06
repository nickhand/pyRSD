from functools import wraps
from collections import OrderedDict
import inspect
import fnmatch

class Property(object):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

class ParameterProperty(Property):
    """
    A subclass of the `property` descriptor to represent
    a required model `Parameter`, i.e, a parameter
    that must be set by the user
    """
    def __set__(self, obj, value):
        """
        Explicitly pass the list of attributes on this 
        """
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)
        
        # clear the cache of any parameters that depend
        # on this attribute
        for dep in self._deps:
            obj._cache.pop(dep, None)

class CachedProperty(Property):
    """
    A subclass of the `property` descriptor to represent an
    an attribute that depends only on other `Parameter` or
    `CachedProperty` values, and can be cached appropriately
    """
    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)
        
        # clear the cache of any parameters that depend
        # on this cached property attribute
        for dep in self._deps:
            obj._cache.pop(dep, None)

class InterpolatedProperty(CachedProperty):
    """
    A subclass of the `property` descriptor to represent an
    an attribute that depends only on other `Parameter` or
    `CachedProperty` values, and can be cached appropriately
    """
    pass
            
def add_metaclass(metaclass):
    """
    Class decorator for creating a class with a metaclass, compatible
    with python 2 and python 3
    
    This is copied from `six.with_metaclass`
    """
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper
    
def CachedModel(cls):
    """ 
    Declares a model that will have its parameters registered and
    cached, keeping track of dependencies
    """
    return add_metaclass(CacheSchema)(cls)


class CacheSchema(type):
    """
    Metaclass to gather all `parameter` and `cached_property`
    attributes from the class, and keep track of dependencies
    for caching purposes
    """
    def __init__(cls, clsname, bases, attrs):
                                    
        # keep track of allowable kwargs of main class
        attrs, varargs, varkw, defaults = inspect.getargspec(cls.__init__)
        cls.allowable_kwargs = set(attrs[1:])
        
        # attach the registry attributes
        cls._cachemap = OrderedDict()
        cls._cached_names = set()
        cls._param_names = set()
        
        # for each class and base classes, track ParameterProperty 
        # and CachedProperty attributes
        classes = inspect.getmro(cls)
        for c in reversed(classes):
            
            # add in allowable keywords from base classes
            if c != cls and hasattr(c, 'allowable_kwargs'):
                cls.allowable_kwargs |= c.allowable_kwargs  
            
            # loop over each attribute
            for name in c.__dict__:                
                value = c.__dict__[name]
                # track cached property
                if isinstance(value, CachedProperty): 
                    cls._cached_names.add(name)
                    cls._cachemap[name] = value._parents
                # track parameters
                elif isinstance(value, ParameterProperty):
                    if c == cls and fnmatch.fnmatch(name, 'use_*_model'):
                        cls.allowable_kwargs.add(name)
                    cls._param_names.add(name)
            
        # invert the cache map
        def invert_cachemap(name, deps):
            """
            Recursively find all cached properties
            that depend on a given parameter
            """
            for param in deps:                
               
                # search classes in order for the attribute
                for cls in classes:
                    f = getattr(cls, param, None)
                    if f is not None: break
                    
                # if a parameter, add to the deps
                if isinstance(f, ParameterProperty):
                    f._deps.add(name)
                # recursively seach all parents of a cached property
                elif isinstance(f, CachedProperty):
                    f._deps.add(name)
                    invert_cachemap(name, f._parents)
                # invalid parent property
                else:
                   raise ValueError("invalid parent property '%s' for cached property '%s'" %(param, name))
                    
        # compute the inverse cache
        for name in cls._cachemap:
            invert_cachemap(name, cls._cachemap[name])
    
@CachedModel
class Cache(object):
    
    def __new__(typ, *args, **kwargs):
        obj = object.__new__(typ, *args, **kwargs)
        obj._cache = {}
        return obj

    def __init__(self):
        pass

def parameter(f):
    """
    Decorator to represent a model parameter that must
    be set by the user
    """
    name = f.__name__
    def _set_property(self, value):
        val = f(self, value)
        setattr(self, name, val)
        return val
        
    @wraps(f)
    def _get_property(self):
        if name not in self.__dict__:
            raise ValueError("required parameter '%s' has not yet been set" %name)
        return self.__dict__[name]
       
    prop = ParameterProperty(_get_property, _set_property, None)
    prop._deps = set() # track things that depend on this parameter
    return prop

def cached_property(*parents):
    """
    Decorator to represent a model parameter will be cached
    and automatically updated if any of its dependencies change
    """
    def cache(f):
        name = f.__name__
        
        @wraps(f)
        def _get_property(self):
            if name not in self._cache:
                self._cache[name] = f(self)
            return self._cache[name]
        
        def _del_property(self):
            self._cache.pop(name, None)
                        
        prop = CachedProperty(_get_property, None, _del_property)
        prop._parents = list(parents) # the dependencies of this property
        prop._deps = set()
        return prop
    return cache

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def interpolated_property(*parents, **kwargs):
    """
    A decorator that represents a cached property that
    is a function of `k`. The cached property that is stored
    is a spline that predicts the function as a function of `k`
    """
    def cache(f):
        name = f.__name__
        
        @wraps(f)
        def _get_property(self):
            
            # the spline isn't in the cache, make the spline
            if name not in self._cache:
                interp_domain = getattr(self, kwargs.get("interp", "k_interp"))
                val = f(self, interp_domain)
                spline_kwargs = getattr(self, 'spline_kwargs', {})
                if isinstance(val, tuple):
                    self._cache[name] = [self.spline(interp_domain, x, **spline_kwargs) for x in val]
                else:
                    self._cache[name] = self.spline(interp_domain, val, **spline_kwargs)
            
            return self._cache[name]
            
        def _del_property(self):
            self._cache.pop(name, None)
                        
        prop = InterpolatedProperty(_get_property, None, None)
        prop._parents = list(parents) # the dependencies of this property
        prop._deps = set()
        return prop
    return cache