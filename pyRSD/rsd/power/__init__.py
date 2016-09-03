from pyRSD.rsd.tools import get_hash_key
import functools

def memoize(f):
    """
    Memoization decorator for power terms
    """
    @functools.wraps(f)
    def wrap(self, k):
        
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        name = "%s.%s" %(self.__class__.__name__, f.__name__)
        hashkey = get_hash_key(name, k)
        if hashkey not in self._cache:
            self._cache[hashkey] = f(self, k)
        return self._cache[hashkey]  
        
    return wrap  

class AngularTerm(object):
    """
    Class to keep track of the different angular terms for each 
    power term.
    """
    base = None
    
    def __init__(self, model):
        self.m = model
        
    def __call__(self, k):
        return self.total(k)
        
    def total(self, k):
        raise NotImplementedError()
        
    def scalar(self, k):
        raise NotImplementedError()
        
    def vector(self, k):
        raise NotImplementedError()
        
    def no_velocity(self, k):
        raise NotImplementedError()
    
    def with_velocity(self, k):
        raise NotImplementedError()


class PowerTerm(object):
    """
    Class to hold the data for each term in the power expansion.
    """    
    def __init__(self, model, **terms):
        for name, term in terms.items():
            t = term(model)
            t.base = self
            setattr(self, name, t)

