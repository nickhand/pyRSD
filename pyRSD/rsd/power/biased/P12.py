from .. import AngularTerm, PowerTerm, memoize
from .P01 import P01_mu2

class P12_mu4(AngularTerm):
    """
    Implement P12[mu4]
    """
    @memoize
    def total(self, k):
        b1, b1_bar  = self.m._ib1, self.m._ib1_bar 
        
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
    
        P01_dm = self.m.P01.mu2(k)
        term1_mu4 = self.m.P12.mu4.no_velocity(k) - 0.5*(self.m.f*k)**2 * 0.5*(sigsq + sigsq_bar) * P01_dm
        term2_mu4 = -0.5*((b1 - 1.) + (b1_bar - 1.))*self.m.f**3*self.m.I03(k)
        term3_mu4 = -0.25*(self.m.f*k)**2 * (sigsq + sigsq_bar) * (P01_mu2(self.m, k, self.m.b2_01_b) - P01_dm)
        
        return term1_mu4 + term2_mu4 + term3_mu4
        
class P12_mu6(AngularTerm):
    """
    Implement P12[mu6]
    """
    @memoize
    def total(self, k):
        b1, b1_bar  = self.m._ib1, self.m._ib1_bar 
        
        Plin = self.m.normed_power_lin(k)
        
        # get the integral attributes
        I21 = self.m.I21(k)
        I30 = self.m.I30(k)
        J20 = self.m.J20(k)
    
        return self.m.f**3 * (I21 - 0.5*(b1+b1_bar)*I30 + 2*k**2*J20*Plin)
        
class P12PowerTerm(PowerTerm):
    """
    The full P12 power term
    """
    def __init__(self, model):
        super(P12PowerTerm, self).__init__(model, mu4=P12_mu4, mu6=P12_mu6)
        
    
