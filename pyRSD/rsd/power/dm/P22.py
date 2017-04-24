from .. import AngularTerm, PowerTerm, memoize

class P22_mu4(AngularTerm):
    """
    Implement P22[mu4]
    """
    @memoize
    def no_velocity(self, k):

        Plin = self.m.normed_power_lin(k)

        # 1-loop or 2-loop terms from <v^2 | v^2 >
        if not self.m.include_2loop:
            toret = 1./16*self.m.f**4 * self.m.I23(k)
        else:
            I23_2loop = self.m.Ivvvv_f23(k)
            toret = 1./16*self.m.f**4 * I23_2loop

        return toret

    @memoize
    def total(self, k):

        if not self.m.include_2loop:
            return self.no_velocity(k)

        # now add in the extra 2 loop terms, if specified
        Plin = self.m.normed_power_lin(k)

        # velocities in units of Mpc/h
        sigma_lin = self.m.sigma_v
        sigma_22  = self.m.sigma_bv2 * self.m.cosmo.h() / (self.m.f*self.m.conformalH)
        sigsq_eff = sigma_lin**2 + sigma_22**2

        J02 = self.m.J02(k)
        J20 = self.m.J20(k)

        # one more 2-loop term for <v^2 | v^2>
        extra_vv_mu4 = (self.m.f*k)**4 * Plin*J02**2

        # term from <v^2 | d v^2>
        extra_vdv_mu4 = -0.5*(self.m.f*k)**2 * sigsq_eff * self.m.P02.mu2.no_velocity(k)

        # 1st term coming from <dv^2 | dv^2>
        extra1_dvdv_mu4 = 0.25*(self.m.f*k)**4 * sigsq_eff**2 * self.m.P00.mu0(k)

        # 2nd term from <dv^2 | dv^2> is convolution of P22_bar and P00
        extra2_dvdv_mu4 = 0.5*(self.m.f*k)**4 * self.m.P00.mu0(k)*self.m.sigmasq_k(k)**2

        # store the extra two loop terms
        extra = extra_vv_mu4 + extra_vdv_mu4 + extra1_dvdv_mu4 + extra2_dvdv_mu4

        return self.no_velocity(k) + extra


class P22_mu6(AngularTerm):
    """
    Implement P22[mu6]
    """
    @memoize
    def no_velocity(self, k):

        # 1-loop or 2-loop terms that don't depend on velocity
        if not self.m.include_2loop:
            toret = 1./8*self.m.f**4 * self.m.I32(k)
        else:
            I32_2loop = self.m.Ivvvv_f32(k)
            toret = 1./8*self.m.f**4 * I32_2loop

        return toret

    @memoize
    def total(self, k):

        if not self.m.include_2loop:
            return self.no_velocity(k)

        # now add in the extra 2 loop terms, if specified
        Plin = self.m.normed_power_lin(k)

        # velocities in units of Mpc/h
        sigma_lin = self.m.sigma_v
        sigma_22  = self.m.sigma_bv2 * self.m.cosmo.h() / (self.m.f*self.m.conformalH)
        sigsq_eff = sigma_lin**2 + sigma_22**2

        J02 = self.m.J02(k)
        J20 = self.m.J20(k)

        # term from <v^2 | d v^2>
        extra_vdv_mu6 = -0.5*(self.m.f*k)**2 * sigsq_eff * self.m.P02.mu4.no_velocity(k)

        # one more 2-loop term for <v^2 | v^2>
        extra_vv_mu6  = 2*(self.m.f*k)**4 * Plin*J02*J20

        extra = extra_vdv_mu6 + extra_vv_mu6
        return self.no_velocity(k) + extra


class P22PowerTerm(PowerTerm):
    """
    The full P22 power term
    """
    def __init__(self, model):
        super(P22PowerTerm, self).__init__(model, mu4=P22_mu4, mu6=P22_mu6)
