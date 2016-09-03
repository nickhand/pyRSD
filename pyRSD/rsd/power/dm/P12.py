from .. import AngularTerm, PowerTerm, memoize
        
class P12_mu4(AngularTerm):
    """
    Implement P12[mu4]
    """        
    @memoize
    def no_velocity(self, k):
        
        Plin = self.m.normed_power_lin(k)
        
        # the necessary integrals 
        I12 = self.m.I12(k)
        I03 = self.m.I03(k)
        J02 = self.m.J02(k)
        
        # do the mu^4 terms that don't depend on velocity
        return self.m.f**3 * (I12 - I03 + 2*k**2*J02*Plin)
        
    @memoize    
    def with_velocity(self, k):
        
        # now do mu^4 terms depending on velocity (velocities in Mpc/h)
        sigma_lin = self.m.sigma_v  
        sigma_12  = self.m.sigma_bv2 * self.m.cosmo.h() / (self.m.f*self.m.conformalH) 
        sigsq_eff = sigma_lin**2 + sigma_12**2
    
        if self.m.include_2loop:
            toret = -0.5*(self.m.f*k)**2 * sigsq_eff * self.m.P01.mu2(k)
        else:
            Plin = self.m.normed_power_lin(k)
            toret = -self.m.f*(self.m.f*k)**2 * sigsq_eff * Plin
        
        return toret
    
    def total(self, k):
        return self.with_velocity(k) + self.no_velocity(k)
        
class P12_mu6(AngularTerm):
    """
    Implement P12[mu6]
    """    
    @memoize    
    def no_velocity(self, k):
        
        Plin = self.m.normed_power_lin(k)
        I21  = self.m.I21(k)
        I30  = self.m.I30(k)
        J20  = self.m.J20(k)
        
        return self.m.f**3 * (I21 - I30 + 2*k**2*J20*Plin)
       
    def total(self, k):
        return self.no_velocity(k)
                    
        
class P12PowerTerm(PowerTerm):
    """
    The full P12 power term
    """
    def __init__(self, model):
        super(P12PowerTerm, self).__init__(model, mu4=P12_mu4, mu6=P12_mu6)
    
