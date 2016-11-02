from . import ConstraintDerivative

class NsBsBConstraint(ConstraintDerivative):
    """
    Class to represent the derivative of the constraint
    for the `NsBsB` parameter
    """
    name = "NsBsB"
    expr = "f1h_sBsB / (fsB**2 * fs**2 * nbar) * (fcB*(1 - fs) - fs*(1-fsB))"

    def __init__(self, x, y):
        super(NsBsBConstraint, self).__init__(x, y)

    def deriv_fs(self, m, pars, k, mu):
        """
        Derivative with respect to `fs`
        """
        f1h_sBsB = pars['f1h_sBsB'].value
        nbar     = pars['nbar'].value
        return f1h_sBsB * (m.fcB*(-2+m.fs) + m.fs*(1-m.fsB)) / (m.fs**3 * m.fsB**2 * nbar)

    def deriv_fcB(self, m, pars, k, mu):
        """
        Derivative with respect to `fcB`
        """
        f1h_sBsB = pars['f1h_sBsB'].value
        nbar     = pars['nbar'].value
        return  (1.-m.fs) * f1h_sBsB / (m.fs**2 * m.fsB**2 * nbar)

    def deriv_f1h_sBsB(self, m, pars, k, mu):
        """
        Derivative with respect to `f1h_cBs`
        """
        nbar     = pars['nbar'].value
        return 1./(m.fsB**2 * m.fs**2 * nbar) * (m.fcB*(1 - m.fs) - m.fs*(1-m.fsB))

    def deriv_nbar(self, m, pars, k, mu):
        """
        Derivative with respect to `nbar`
        """
        f1h_sBsB = pars['f1h_sBsB'].value
        nbar     = pars['nbar'].value
        return -f1h_sBsB /(m.fsB**2 * m.fs**2 * nbar**2) * (m.fcB*(1 - m.fs) - m.fs*(1-m.fsB))
        
                
        
        
    
