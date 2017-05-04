from .. import numpy
from ._interpolate import InterpolationDomainError

import functools
from collections import OrderedDict
import inspect
import fnmatch
from six import add_metaclass, PY3, string_types

try:
    from functools import lru_cache
except ImportError:
    from pyRSD.extern.backports.lru_cache import lru_cache

import types

if not PY3:
    def _pickle_method(m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    import copyreg
    copyreg.pickle(types.MethodType, _pickle_method)

def doublewrap(f):
    """
    A decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """
    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            # actual decorated function
            return f(args[0], **kwargs)
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec

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
        self.fset(obj, value, deps=self._deps)

class CachedProperty(Property):
    """
    A subclass of the `property` descriptor to represent an
    an attribute that depends only on other `Parameter` or
    `CachedProperty` values, and can be cached appropriately
    """
    __cache__ = True

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

        # clear the cache of any parameters that depend
        # on this cached property attribute
        for dep in self._deps:
            obj._cache.pop(dep, None)

class CacheSchema(type):
    """
    Metaclass to gather all `parameter` and `cached_property`
    attributes from the class, and keep track of dependencies
    for caching purposes
    """
    def __init__(cls, clsname, bases, attrs):

        # keep track of allowable kwargs of main class
        cls.allowable_kwargs = set()
        if PY3:
            sig = inspect.signature(cls.__init__)
            for name, p in sig.parameters.items():
                if p.default != inspect.Parameter.empty:
                    cls.allowable_kwargs.add(name)
        else:
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
                if getattr(value, '__cache__', False):
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
                elif isinstance(f, CachedProperty) or getattr(f, '__cache__', False):
                    f._deps.add(name)
                    invert_cachemap(name, f._parents)
                # invalid parent property
                else:
                    if hasattr(f, '_deps'):
                        f._deps.add(name)
                    else:
                        raise ValueError("invalid parent property '%s' for cached property '%s'" %(param, name))

        # compute the inverse cache
        for name in cls._cachemap:
            invert_cachemap(name, cls._cachemap[name])

@add_metaclass(CacheSchema)
class Cache(object):
    """
    The main class to do handle caching of parameters; this is the
    class that should serve as the base class
    """
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._cache = {}
        return obj

    def __init__(self, *args, **kwargs):
        super(Cache, self).__init__(*args, **kwargs)

def obj_eq(new_val, old_val):
    """
    Test the equality of an old and new value
    """
    equal = False
    try:
        if old_val is not None:
            try:
                if numpy.isscalar(new_val) and numpy.isscalar(old_val):
                    equal = new_val == old_val
                else:
                    equal = numpy.allclose(new_val, old_val)
            except:
                return new_val == old_val
    except:
        pass
    return equal

@doublewrap
def parameter(f, default=None):
    """
    Decorator to represent a model parameter that must
    be set by the user
    """
    name = f.__name__
    _name = '__'+name
    def _set_property(self, value, deps=[]):
        val = f(self, value)
        try:
            old_val = getattr(self, _name)
            doset = False
        except AttributeError:
            old_val = None
            doset = True

        if doset or not obj_eq(val, old_val):
            setattr(self, _name, val)

            # clear the cache of any parameters that depend
            # on this attribute
            for dep in deps:
                self._cache.pop(dep, None)
        return val

    @functools.wraps(f)
    def _get_property(self):
        if _name not in self.__dict__:

            if default is not None:
                if isinstance(default, string_types) and hasattr(self, default):
                    val = getattr(self, default)
                    return val
                else:
                    return f(self, default)
            else:
                raise ValueError("required parameter '%s' has not yet been set" %name)
        else:
            return self.__dict__[_name]

    def _del_property(self):
        if _name in self.__dict__:
            self.__dict__.pop(_name)
        else:
            raise ValueError("cannot delete attribue '%s'" %name)

    prop = ParameterProperty(_get_property, _set_property, _del_property)
    prop._deps = set() # track things that depend on this parameter
    return prop

def cached_property(*parents, **kws):
    """
    Decorator to represent a model parameter will be cached
    and automatically updated if any of its dependencies change
    """
    _lru_cache = kws.pop('lru_cache', False)
    maxsize = kws.pop('maxsize', 128)

    def cache(f):
        name = f.__name__

        @functools.wraps(f)
        def _get_property(self):

            # check for user overrides
            if name in getattr(self, "_cache_overrides", {}):
                return self._cache_overrides[name]

            # add to cache
            if name not in self._cache:
                val = f(self)
                if _lru_cache and callable(val):
                    val = lru_cache(maxsize=maxsize)(val)
                self._cache[name] = val

            # return the cached value
            return self._cache[name]

        def _del_property(self):
            self._cache.pop(name, None)

        prop = CachedProperty(_get_property, None, _del_property)
        prop._parents = list(parents) # the dependencies of this property
        prop._deps = set()

        prop._lru_cache = _lru_cache
        prop._maxsize   = maxsize

        return prop
    return cache

class InterpolatedFunction(object):
    """
    A callable class (as a function of wavenumber) that will
    either evaluate a spline or evaluate the underlying function,
    if a domain error occurs
    """
    def __init__(self, spline, function, name):
        self.spline   = spline
        self.function = function
        self.name     = name

    def __call__(self, k, derivative=False):
        try:
            if isinstance(self.spline, list):
                if not derivative:
                    return [spl(k) for spl in self.spline]
                else:
                    return [spl.derivative()(k) for spl in self.spline]
            else:
                if not derivative:
                    return self.spline(k)
                else:
                    return self.spline.derivative()(k)
        except InterpolationDomainError:

            # print a warning
            import warnings
            warnings.warn("evaluating interpolated function %s outside of domain...will be slow!!" %(self.name))

            # return spline results in bounds, and call the function out of bounds
            if numpy.isscalar(k) or numpy.ndim(k) == 0:
                return self.function(k)[0]
            else:
                k = numpy.asarray(k)
                out_of_bounds = self.spline._check_bounds(k)
                toret = numpy.zeros_like(k)
                toret[out_of_bounds] = self.function(k[out_of_bounds])
                toret[~out_of_bounds] = self.spline(k[~out_of_bounds])
                return toret
        except:
            raise

def interpolated_function(*parents, **kwargs):
    """
    A decorator that represents a cached property that
    is a function of `k`. The cached property that is stored
    is a spline that predicts the function as a function of `k`
    """
    def wrapper(f):
        name = f.__name__

        @functools.wraps(f)
        def wrapped(self, *args, **kws):
            """
            If `ignore_cache` is True, force an evaluation
            of the decorated function
            """
            ignore_cache = kws.pop('ignore_cache', False)
            if ignore_cache:
                return f(self, *args)

            # the spline isn't in the cache, make the spline
            if name not in self._cache:

                # make the spline
                interp_domain = getattr(self, kwargs.get("interp", "k_interp"))
                val = f(self, interp_domain)
                spline_kwargs = getattr(self, 'spline_kwargs', {})

                # the function that will explicitly evaluate the original function
                g = wrapped.__get__(self, self.__class__)

                # tuple of splines
                if isinstance(val, tuple):
                    splines = [self.spline(interp_domain, x, **spline_kwargs) for x in val]
                    self._cache[name] = InterpolatedFunction(splines, g, name)
                # single spline
                else:
                    spl = self.spline(interp_domain, val, **spline_kwargs)
                    self._cache[name] = InterpolatedFunction(spl, g, name)

            return self._cache[name](*args, **kws)

        # store the meta information about this property
        wrapped._parents = list(parents) # the dependencies of this property
        wrapped._deps = set()
        wrapped.__cache__ = True

        return wrapped

    return wrapper
