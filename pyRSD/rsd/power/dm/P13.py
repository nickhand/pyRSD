from .. import AngularTerm, PowerTerm, memoize

class P13_mu4(AngularTerm):
    """
    Implement P13[mu4]
    """
    @memoize
    def with_velocity(self, k):
        
        if not self.m.include_2loop:
            return 0.
                 
        # compute velocity weighting in Mpc/h
        sigma_lin = self.m.sigma_v 
        sigma_13_v  = self.m.sigma_bv2 * self.m.cosmo.h() / (self.m.f*self.m.conformalH) 
        sigsq_eff_vector = sigma_lin**2 + sigma_13_v**2
        
        A = -(self.m.f*k)**2
        return A*sigsq_eff_vector*self.m.P11.mu2(k)
        
    def total(self, k):
        return self.with_velocity(k)
        
class P13_mu6(AngularTerm):
    """
    Implement P13[mu6]
    """
    @memoize
    def with_velocity(self, k):
                  
        # compute velocity weighting in Mpc/h
        sigma_lin = self.m.sigma_v         
        sigma_13_s  = self.m.sigma_v2 * self.m.cosmo.h() / (self.m.f*self.m.conformalH) 
        sigsq_eff_scalar = sigma_lin**2 + sigma_13_s**2
        
        # mu^6 velocity terms at 1 or 2 loop
        if self.m.include_2loop:
            A = -(self.m.f*k)**2
            toret = A*sigsq_eff_scalar*self.m.P11.mu4(k)
        else:
            Plin = self.normed_power_lin(k)
            toret = -self.m.f**2 *(self.m.f*k)**2 * sigsq_eff_scalar*Plin
            
        return toret
        
    def total(self, k):
        return self.with_velocity(k)
        
class P13PowerTerm(PowerTerm):
    """
    The full P13 power term
    """
    def __init__(self, model):
        super(P13PowerTerm, self).__init__(model, mu4=P13_mu4, mu6=P13_mu6)
    