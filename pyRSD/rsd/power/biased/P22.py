from .. import AngularTerm, PowerTerm, memoize
from .P00 import Phh_mu0

class P22_mu4(AngularTerm):
    """
    Implement P22[mu4]
    """
    @memoize
    def total(self, k):
        
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        b2_00, b2_00_bar = self.m.b2_00_d(b1), self.m.b2_00_d(b1_bar)
        
        # velocities in units of Mpc/h
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2

        # 1-loop P22bar
        term1 = self.m.P22.mu4.no_velocity(k)
    
        # add convolution to P22bar
        term2 = 0.5*(self.m.f*k)**4 * (b1*b1_bar * self.m.Pdd(k)) * self.m.sigmasq_k(k)**2
    
        # b1 * P02_bar
        term3 = -0.25*(k*self.m.f)**2 * (sigsq + sigsq_bar) * ( 0.5*(b1 + b1_bar)*self.m.P02.mu2.no_velocity(k))
    
        # sigma^4 x P00_ss
        term4 = 0.125*(k*self.m.f)**4 * (sigsq**2 + sigsq_bar**2) * Phh_mu0(self.m, k, self.m.b2_00_d)
    
        return term1 + term2 + term3 + term4
        
class P22_mu6(AngularTerm):
    """
    Implement P22[mu6]
    """
    @memoize
    def total(self, k):
        
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
        
        term1 = self.m.P22.mu6.no_velocity(k)
        term2 = -0.25*(k*self.m.f)**2 * (sigsq + sigsq_bar) * (0.5*(b1 + b1_bar)*self.m.P02.mu4.no_velocity(k))
        return term1 + term2
        
class P22PowerTerm(PowerTerm):
    """
    The full P22 power term
    """
    def __init__(self, model):
        super(P22PowerTerm, self).__init__(model, mu4=P22_mu4, mu6=P22_mu6)
        
    
