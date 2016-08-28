import abc 

class HashableMeta(abc.ABCMeta):
    """
    Metaclass to implement a custom `hash` of the class
    definition based on the :attr:`name` and :attr:`expr`
    """
    def __hash__(cls):
        expr = "".join(cls.expr.split())
        return hash((cls.name, expr))
        
        
class ConstraintDerivative(object, metaclass=HashableMeta):
    """
    Class to represent the derivative of a constraint function
    """
    @classmethod
    def registry(cls):
        """
        Return the registered subclass
        """
        d = {}
        for subclass in cls.__subclasses__():
            d[hash(subclass)] = subclass
            
        return d
        
    def __init__(self, x, y):
        """
        Parameters
        ----------
        x : Parameter
            the parameter that we are taking the derivative of
        y : Parameter
            the parameter that we are taking the derivative with respect to
        """
        self.x = x
        self.y = y
        
        expr = "".join(x.expr.split())
        if expr != "".join(self.expr.split()):
            raise ValueError() 
        
        if self.x.name != self.__class__.name:
            raise ValueError()
                    
    def __call__(self, *args):
        """
        Evaluate the derivative
        
        Parameters
        ----------
        m : GalaxySpectrum
            the model instance
        pars : ParameterSet
            the theory parameters
        k : array_like
            the array of `k` values to evaluate the derivative at
        mu : array_like
            the array of `mu` values to evaluate the derivative at
        """
        func_name = 'deriv_'+self.y.name
        if not hasattr(self, func_name):
            tup = (func_name, self.x.name, self.y.name)
            raise NotImplementedError("function '%s' is needed to compute d%s/d%s" %tup)
            
        return getattr(self, func_name)(*args)

        
def get_constraint_derivative(par, *args):
    """
    Return the constraint derivative class
    """
    name = par.name
    if par.expr is None:
        raise ValueError("no constraint expression for parameter %s" %str(par))
    expr = "".join(par.expr.split())
    
    key = (name, expr)
    registry =  ConstraintDerivative.registry()
    if hash(key) not in registry:
        raise ValueError("no registered constraint derivative for %s" %str(key))
        
    return registry[hash(key)](par, *args)
    
from .fcB   import fcBConstraint
from .fso   import fsoConstraint
from .NcBs  import NcBsConstraint
from .NsBsB import NsBsBConstraint

__all__ = [ 'get_constraint_derivative',
            'fcBConstraint', 
            'fsoConstraint',
            'NcBsConstraint', 
            'NsBsBConstraint'
          ]