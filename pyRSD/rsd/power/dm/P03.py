from .. import AngularTerm, PowerTerm, memoize

class P03_mu4(AngularTerm):
    """
    Implement P03[mu4]
    """
    @memoize
    def with_velocity(self, k):
                  
        # only terms depending on velocity here (velocities in Mpc/h)
        sigma_lin = self.m.sigma_v 
        sigma_03  = self.m.sigma_v2 * self.m.cosmo.h() / (self.m.f*self.m.conformalH)
        sigsq_eff = sigma_lin**2 + sigma_03**2

        # either 1 or 2 loop quantities
        if self.m.include_2loop:
            toret = -0.5*(self.m.f*k)**2 * sigsq_eff * self.m.P01.mu2(k)
        else:
            Plin = self.m.normed_power_lin(k)
            toret = -self.m.f*(self.m.f*k)**2 *sigsq_eff*Plin
    
        return toret
        
    def total(self, k):
        return self.with_velocity(k)
        
class P03PowerTerm(PowerTerm):
    """
    The full P03 power term
    """
    def __init__(self, model):
        super(P03PowerTerm, self).__init__(model, mu4=P03_mu4)
    