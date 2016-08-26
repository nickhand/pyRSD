from . import ConstraintDerivative

class NcBsConstraint(ConstraintDerivative):
    """
    Class to represent the derivative of the constraint
    for the `NcBs` parameter
    """
    name = "NcBs"    
    expr = "f1h_cBs / (fcB*(1 - fs)*nbar)"
    
    def __init__(self, x, y):
        super(NcBsConstraint, self).__init__(x, y)
        
    def deriv_fs(self, m, pars, k, mu):
        """
        Derivative with respect to `fs`
        """
        f1h_cBs = pars['f1h_cBs'].value
        nbar    = pars['nbar'].value
        return f1h_cBs / (m.fcB * (1 - m.fs)**2 * nbar)
        
    def deriv_fcB(self, m, pars, k, mu):
        """
        Derivative with respect to `fcB`
        """
        f1h_cBs = pars['f1h_cBs'].value
        nbar    = pars['nbar'].value
        return -f1h_cBs / (m.fcB**2 * (1 - m.fs)* nbar)
        
    def deriv_f1h_cBs(self, m, pars, k, mu):
        """
        Derivative with respect to `f1h_cBs`
        """
        return 1. / (m.fcB * (1 - m.fs)* pars['nbar'].value)
        
    def deriv_nbar(self, m, pars, k, mu):
        """
        Derivative with respect to `nbar`
        """
        f1h_cBs = pars['f1h_cBs'].value
        nbar    = pars['nbar'].value
        return -f1h_cBs / (m.fcB * (1 - m.fs)* nbar**2)
        
                
        
        
    