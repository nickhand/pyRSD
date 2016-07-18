"""
Module to implement nonlinear biases as a function of 
linear bias
"""
from ._cache import parameter
from .. import numpy as np, data as sim_data

class NonlinearBiasPolynomial(object):
    """
    A class to return the nonlinear bias, using a polynomial
    as a function of linear bias
    
    This class is a callable, and returns a ``numpy.poly1d``
    callable instance
    """
    def __init__(self, name):
        self.__name__ = name
        
    def __call__(self, model, val):
        return np.poly1d(val)


class WithPolynomialBiasParameters(type):
    """
    Metaclass that attaches a `parameter` instance for each of the
    nonlinear biasing parameters that we want to use
    """  
    def __init__(cls, name, bases, dict):
        
        # add a "parameter" for each of the nonlinear biasing terms
        # and initialize to default coefficients
        for name in cls.nonlinear_biases:
            if 'b2_00' in name:
                defaults = sim_data.nonlinear_bias_data('b2_00', name)
            elif 'b2_01' in name:
                defaults = sim_data.nonlinear_bias_data('b2_01', name)
            else:
                raise ValueError("only nonlinear biases are 'b2_00' or 'b2_01'")
            param = parameter(NonlinearBiasPolynomial(name), default=defaults)
            setattr(cls, name, param)
            
        super(WithPolynomialBiasParameters, cls).__init__(name, bases, dict)

class NonlinearBiasingMixin(object):
    """
    A mix-in class to implement the relevant nonlinear biasing 
    functions, which uses a polynomial as a function of linear bias
    
    This class will have "parameter" objects (added dynamically, via
    the metaclass) for each of the names in `nonlinear_biases`
    
    The nonlinear biasing parameters should be set to their polynomial
    coefficients, and when accessed, the parameters will return
    `numpy.poly1d` instances, which can be called with the appropriate
    linear bias parameter
    """
    nonlinear_biases = ['b2_00_a', 'b2_00_b', 'b2_00_c', 'b2_00_d', 'b2_01_a', 'b2_01_b']
    __metaclass__ = WithPolynomialBiasParameters