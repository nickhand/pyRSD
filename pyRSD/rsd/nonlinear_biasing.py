"""
Module to implement nonlinear biases as a function of 
linear bias
"""
from ._cache import parameter, cached_property
from .. import numpy as np, data as sim_data

# class NonlinearBiasPolynomial(object):
#     """
#     A class to return the nonlinear bias, using a polynomial
#     as a function of linear bias
#
#     This class is a callable, and returns a ``numpy.poly1d``
#     callable instance
#     """
#     def __init__(self, name):
#         self.__name__ = name
#
#     def __call__(self, model, val):
#         return np.poly1d(val)
#
#
# class WithPolynomialBiasParameters(type):
#     """
#     Metaclass that attaches a `parameter` instance for each of the
#     nonlinear biasing parameters that we want to use
#     """
#     def __init__(cls, name, bases, dict):
#
#         # add a "parameter" for each of the nonlinear biasing terms
#         # and initialize to default coefficients
#         for name in cls.nonlinear_biases:
#             if 'b2_00' in name:
#                 defaults = sim_data.nonlinear_bias_data('b2_00', name)
#             elif 'b2_01' in name:
#                 defaults = sim_data.nonlinear_bias_data('b2_01', name)
#             else:
#                 raise ValueError("only nonlinear biases are 'b2_00' or 'b2_01'")
#             param = parameter(NonlinearBiasPolynomial(name), default=defaults)
#             setattr(cls, name, param)
#
#         super(WithPolynomialBiasParameters, cls).__init__(name, bases, dict)

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
    #__metaclass__ = WithPolynomialBiasParameters
    
    def __init__(self):
        
        for name in self.nonlinear_biases:
            
            names = [name+'__'+str(i) for i in reversed(range(5))]
            if 'b2_00' in name:
                defaults = sim_data.nonlinear_bias_data('b2_00', name)
            elif 'b2_01' in name:
                defaults = sim_data.nonlinear_bias_data('b2_01', name)
            else:
                raise ValueError("only nonlinear biases are 'b2_00' or 'b2_01'")
                
            # set the defaults
            for i, d in enumerate(defaults):
                print names[i]
                if hasattr(self.__class__, names[i]):
                    setattr(self, names[i], d)
    
    #--------------------------------------------------------------------------
    # b2_00_a
    #--------------------------------------------------------------------------
    @parameter
    def b2_00_a__0(self, val):
        return val
        
    @parameter
    def b2_00_a__2(self, val):
        return val
    
    @parameter
    def b2_00_a__4(self, val):
        return val
        
    @cached_property("b2_00_a__0", "b2_00_a__2", "b2_00_a__4")
    def b2_00_a(self):
        coeffs = [self.b2_00_a__4, 0., self.b2_00_a__2, 0., self.b2_00_a__0]
        return np.poly1d(coeffs)
        
    #--------------------------------------------------------------------------
    # b2_00_b
    #--------------------------------------------------------------------------
    @parameter
    def b2_00_b__0(self, val):
        return val
        
    @parameter
    def b2_00_b__2(self, val):
        return val
    
    @parameter
    def b2_00_b__4(self, val):
        return val
        
    @cached_property("b2_00_b__0", "b2_00_b__2", "b2_00_b__4")
    def b2_00_b(self):
        coeffs = [self.b2_00_b__4, 0., self.b2_00_b__2, 0., self.b2_00_b__0]
        return np.poly1d(coeffs)
    
    #--------------------------------------------------------------------------
    # b2_00_c
    #--------------------------------------------------------------------------
    @parameter
    def b2_00_c__0(self, val):
        return val
        
    @parameter
    def b2_00_c__2(self, val):
        return val
    
    @parameter
    def b2_00_c__4(self, val):
        return val
        
    @cached_property("b2_00_c__0", "b2_00_c__2", "b2_00_c__4")
    def b2_00_c(self):
        coeffs = [self.b2_00_c__4, 0., self.b2_00_c__2, 0., self.b2_00_c__0]
        return np.poly1d(coeffs)
    
    #--------------------------------------------------------------------------
    # b2_00_d
    #--------------------------------------------------------------------------
    @parameter
    def b2_00_d__0(self, val):
        return val
        
    @parameter
    def b2_00_d__2(self, val):
        return val
    
    @parameter
    def b2_00_d__4(self, val):
        return val
        
    @cached_property("b2_00_d__0", "b2_00_d__2", "b2_00_d__4")
    def b2_00_d(self):
        coeffs = [self.b2_00_d__4, 0., self.b2_00_d__2, 0., self.b2_00_d__0]
        return np.poly1d(coeffs)
        
    #--------------------------------------------------------------------------
    # b2_01_a
    #--------------------------------------------------------------------------
    @parameter
    def b2_01_a__0(self, val):
        return val
        
    @parameter
    def b2_01_a__1(self, val):
        return val
    
    @parameter
    def b2_01_a__2(self, val):
        return val
        
    @cached_property("b2_01_a__0", "b2_01_a__1", "b2_01_a__2")
    def b2_01_a(self):
        coeffs = [0., 0., self.b2_01_a__2, self.b2_01_a__1, self.b2_01_a__0]
        return np.poly1d(coeffs)
        
    #--------------------------------------------------------------------------
    # b2_01_b
    #--------------------------------------------------------------------------
    @parameter
    def b2_01_b__0(self, val):
        return val
        
    @parameter
    def b2_01_b__1(self, val):
        return val
    
    @parameter
    def b2_01_b__2(self, val):
        return val
        
    @cached_property("b2_01_b__0", "b2_01_b__1", "b2_01_b__2")
    def b2_01_b(self):
        coeffs = [0., 0., self.b2_01_b__2, self.b2_01_b__1, self.b2_01_b__0]
        return np.poly1d(coeffs)
    