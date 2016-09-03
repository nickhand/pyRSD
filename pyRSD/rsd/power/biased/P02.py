from .. import AngularTerm, PowerTerm, memoize
from .P00 import Phh_mu0
  
class P02_mu2(AngularTerm):
    """
    Implement P02[mu2]
    """
    @memoize
    def total(self, k):
        
        # the biasing
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        bs, bs_bar = self.m.bs, self.m.bs_bar
        b2_00, b2_00_bar = self.m.b2_00_c(b1), self.m.b2_00_c(b1_bar)
        
        # the velocities
        sigsq     = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
    
        # the PT integrals
        K20_a  = self.m.K20_a(k)
        K20s_a = self.m.K20s_a(k)
    
        # the individual terms
        term1_mu2 = 0.5*(b1 + b1_bar) * self.m.P02.mu2.no_velocity(k)
        term2_mu2 = -0.5*(self.m.f*k)**2 * (sigsq + sigsq_bar) * Phh_mu0(self.m, k, self.m.b2_00_c)
        term3_mu2 = 0.5*self.m.f**2 * ( (b2_00 + b2_00_bar)*K20_a + (bs + bs_bar)*K20s_a )
        
        return term1_mu2 + term2_mu2 + term3_mu2
        
class P02_mu4(AngularTerm):
    """
    Implement P02[mu4]
    """
    @memoize
    def total(self, k):
        
        # the biasing
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        bs, bs_bar = self.m.bs, self.m.bs_bar
        b2_00, b2_00_bar = self.m.b2_00_b(b1), self.m.b2_00_b(b1_bar)
                            
        # the PT integrals
        K20_b = self.m.K20_b(k)
        K20s_b = self.m.K20s_b(k)
    
        term1_mu4 = 0.5*(b1 + b1_bar) * self.m.P02.mu4.no_velocity(k)
        term2_mu4 = 0.5 * self.m.f**2 * ( (b2_00 + b2_00_bar)*K20_b + (bs + bs_bar)*K20s_b )
        return term1_mu4 + term2_mu4


class P02PowerTerm(PowerTerm):
    """
    The full P02 power term
    """
    def __init__(self, model):
        super(P02PowerTerm, self).__init__(model, mu2=P02_mu2, mu4=P02_mu4)
        
    
