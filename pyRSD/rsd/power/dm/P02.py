from .. import AngularTerm, PowerTerm, memoize
        
class P02_mu2(AngularTerm):
    """
    Implement P02[mu2]
    """
    @memoize
    def no_velocity(self, k):
        
        # linear power
        Plin = self.m.normed_power_lin(k)
        
        # the necessary integrals 
        I02 = self.m.I02(k)
        J02 = self.m.J02(k)

        # the mu^2 no velocity terms
        return self.m.f**2 * (I02 + 2.*k**2*J02*Plin)
        
    @memoize
    def with_velocity(self, k):
        
        # linear power
        Plin = self.m.normed_power_lin(k)
        
        # the mu^2 terms depending on velocity (velocities in Mpc/h)
        sigma_lin = self.m.sigma_v
        sigma_02  = self.m.sigma_bv2 * self.m.cosmo.h() / (self.m.f*self.m.conformalH)
        sigsq_eff = sigma_lin**2 + sigma_02**2

        if self.m.include_2loop:
            toret = -(self.m.f*k)**2 * sigsq_eff*self.m.P00.mu0(k)
        else:
            toret = -(self.m.f*k)**2 * sigsq_eff*Plin
        
        return toret
        
    def total(self, k):
         return self.with_velocity(k) + self.no_velocity(k)

class P02_mu4(AngularTerm):
    """
    Implement P02[mu4]
    """ 
    @memoize       
    def no_velocity(self, k):
        
        Plin = self.m.normed_power_lin(k)
        I20 = self.m.I20(k)
        J20 = self.m.J20(k)
        return self.m.f**2 * (I20 + 2*k**2*J20*Plin)
        
    def total(self, k):
        return self.no_velocity(k)
                    
        
class P02PowerTerm(PowerTerm):
    """
    The full P02 power term
    """
    def __init__(self, model):
        super(P02PowerTerm, self).__init__(model, mu2=P02_mu2, mu4=P02_mu4)
    