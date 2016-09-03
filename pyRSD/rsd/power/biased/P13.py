from .. import AngularTerm, PowerTerm, memoize

class P13_mu4(AngularTerm):
    """
    Implement P13[mu4]
    """
    @memoize
    def total(self, k):
        b1, b1_bar  = self.m._ib1, self.m._ib1_bar 
        
        # velocities
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
    
        A = -(self.m.f*k)**2
        return 0.5*A*(sigsq + sigsq_bar)*self.m.P11_ss.mu2(k)
        
class P13_mu6(AngularTerm):
    """
    Implement P13[mu6]
    """
    @memoize
    def total(self, k):
        b1, b1_bar  = self.m._ib1, self.m._ib1_bar 
        
        # velocities
        sigsq = self.m.sigmav_halo**2
        sigsq_bar = self.m.sigmav_halo_bar**2
    
        A = -(self.m.f*k)**2
        return 0.5*A*(sigsq + sigsq_bar)*self.m.P11_ss.mu4(k)
        
class P13PowerTerm(PowerTerm):
    """
    The full P13 power term
    """
    def __init__(self, model):
        super(P13PowerTerm, self).__init__(model, mu4=P13_mu4, mu6=P13_mu6)
        
    
