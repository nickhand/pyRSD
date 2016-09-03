from .. import AngularTerm, PowerTerm, memoize
from .P00 import Phh_mu0

class P04_mu4(AngularTerm):
    """
    Implement P04[mu4]
    """
    @memoize
    def total(self, k):
        
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        b2_00, b2_00_bar = self.m.b2_00_d(b1), self.m.b2_00_d(b1_bar)
        
        # velocities in Mpc/h
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
    
        # contribution from P02[mu^2]
        term1 = -0.125*(b1 + b1_bar)*(self.m.f*k)**2 * (sigsq + sigsq_bar) * self.m.P02.mu2.no_velocity(k)
    
        # contribution here from P00_ss * vel^4
        A = (1./12)*(self.m.f*k)**4 * Phh_mu0(self.m, k, self.m.b2_00_d)
        term2 = A*(3.*0.5*(sigsq**2 + sigsq_bar**2) + self.m.velocity_kurtosis)
    
        return term1 + term2
        
class P04_mu6(AngularTerm):
    """
    Implement P04[mu6]
    """
    @memoize
    def total(self, k):
        
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
        
        return -0.125*(b1 + b1_bar)*(self.m.f*k)**2 * (sigsq + sigsq_bar) * self.m.P02.mu4.no_velocity(k)
        
class P04PowerTerm(PowerTerm):
    """
    The full P04 power term
    """
    def __init__(self, model):
        super(P04PowerTerm, self).__init__(model, mu4=P04_mu4, mu6=P04_mu6)
        
    
