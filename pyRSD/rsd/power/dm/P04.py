from .. import AngularTerm, PowerTerm, memoize

class P04_mu4(AngularTerm):
    """
    Implement P04[mu4]
    """
    @memoize
    def with_velocity(self, k):
        
        if not self.m.include_2loop:
            return 0.
             
        # compute the relevant small-scale + linear velocities in Mpc/h
        sigma_lin = self.m.sigma_v 
        sigma_04  = self.m.sigma_bv4 * self.m.cosmo.h() / (self.m.f*self.m.conformalH) 
        sigsq_eff = sigma_lin**2 + sigma_04**2
        
        # do P04 mu^4 terms depending on velocity
        P04_vel_mu4_1 = -0.5*(self.m.f*k)**2 * sigsq_eff * self.m.P02.mu2.no_velocity(k)
        P04_vel_mu4_2 = 0.25*(self.m.f*k)**4 * sigsq_eff**2 * self.m.P00.mu0(k)
        return P04_vel_mu4_1 + P04_vel_mu4_2
        
    @memoize
    def no_velocity(self, k):
        
        if not self.m.include_2loop:
            return 0.
        
        return 1./12.*(self.m.f*k)**4 * self.m.P00.mu0(k)*self.m.velocity_kurtosis
        
    def total(self, k):
        return self.no_velocity(k) + self.with_velocity(k)
        
class P04_mu6(AngularTerm):
    """
    Implement P04[mu6]
    """
    @memoize
    def with_velocity(self, k):
        
        if not self.m.include_2loop:
            return 0.
             
        # compute the relevant small-scale + linear velocities in Mpc/h
        sigma_lin = self.m.sigma_v 
        sigma_04  = self.m.sigma_bv4 * self.m.cosmo.h() / (self.m.f*self.m.conformalH) 
        sigsq_eff = sigma_lin**2 + sigma_04**2

        return -0.5*(self.m.f*k)**2 * sigsq_eff * self.m.P02.mu4.no_velocity(k)
        
    def total(self, k):
        return self.with_velocity(k)
        
class P04PowerTerm(PowerTerm):
    """
    The full P04 power term
    """
    def __init__(self, model):
        super(P04PowerTerm, self).__init__(model, mu4=P04_mu4, mu6=P04_mu6)
    