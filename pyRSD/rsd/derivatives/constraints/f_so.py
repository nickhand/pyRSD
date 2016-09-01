from . import ConstraintDerivative
import numpy

class f_soConstraint(ConstraintDerivative):
    """
    Class to represent the derivative of the constraint
    for the `f_so` parameter
    """
    name = "f_so"    
    expr = "10**log10_fso"
    
    def __init__(self, x, y):
        super(f_soConstraint, self).__init__(x, y)
        
    def deriv_log10_fso(self, m, pars, k, mu):
        """
        Derivative with respect to `log10_fso`
        """
        log10_fso = pars['log10_fso'].value
        return 10**log10_fso * numpy.log(10.)
        
        