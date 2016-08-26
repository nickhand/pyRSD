from . import ConstraintDerivative

class fcBConstraint(ConstraintDerivative):
    """
    Class to represent the derivative of the constraint
    for the `fcB` parameter
    """
    name = "fcB"    
    expr = "fs / (1 - fs) * (1 + fsB*(1./Nsat_mult - 1))"
    
    def __init__(self, x, y):
        super(fcBConstraint, self).__init__(x, y)
        
    def deriv_fs(self, m, pars, k, mu):
        """
        Derivative with respect to `fs`
        """
        Nsat_mult = pars['Nsat_mult'].value 
        return (m.fsB - (-1+m.fsB)*Nsat_mult) / (1-m.fs)**2 / Nsat_mult
        
    def deriv_fsB(self, m, pars, k, mu):
        """
        Derivative with respect to `fsB`
        """
        Nsat_mult = pars['Nsat_mult'].value 
        return m.fs/(1.-m.fs) * (1./Nsat_mult - 1.)
        
    def deriv_Nsat_mult(self, m, pars, k, mu):
        """
        Derivative with respect to `Nsat_mult`
        """
        Nsat_mult = pars['Nsat_mult'].value 
        return -m.fs/(1-m.fs) * m.fsB / Nsat_mult**2