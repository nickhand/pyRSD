"""
The derivative of the FOG kernels
"""
from . import PgalDerivative
import numpy


def get_fog_derivative(name, x):
    """
    Get the FOG kernel derivative from the string name
    """
    return globals()[name](x)

def modified_lorentzian(x):
    """
    The partial derivative of :math:`G(x)` with respect to `x`, 
    where
    
    .. math::
    
        G(x) = 1 / (1 + 0.5 x^2)^2
    """
    return -2*x / (1. + 0.5*x**2)**3
        
def lorentzian(x):
    """
    The partial derivative of :math:`G(x)` with respect to `x`, 
    where
    
    .. math::
    
        G(x) = 1 / (1 + 0.5 x^2)
    """
    return -x / (1. + 0.5*x**2)**2
    

def gaussian(x):
    """
    The partial derivative of :math:`G(x)` with respect to `x`, 
    where
    
    .. math::
    
        G(x) = exp[-0.5 x^2]
    """
    return -x * numpy.exp(-0.5 * x**2)
    
