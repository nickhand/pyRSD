from .. import AngularTerm, PowerTerm, memoize
from .P01 import P01_mu2

class P03_mu4(AngularTerm):
    """
    Implement P03[mu4]
    """
    @memoize
    def total(self, k):
        
        b1, b1_bar  = self.m._ib1, self.m._ib1_bar 
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
        return -0.25*(self.m.f*k)**2 * (sigsq + sigsq_bar) * P01_mu2(self.m, k, self.m.b2_01_b)


class P03PowerTerm(PowerTerm):
    """
    The full P03 power term
    """
    def __init__(self, model):
        super(P03PowerTerm, self).__init__(model, mu4=P03_mu4)
        
    
