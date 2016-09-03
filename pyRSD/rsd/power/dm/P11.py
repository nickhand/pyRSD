from .. import AngularTerm, PowerTerm, memoize

class P11_mu2(AngularTerm):
    """
    Implement P11[mu2]
    """
    @memoize
    def vector(self, k):
        
        # check and return any user-loaded values
        if self.m.P11_mu2_loaded:
            toret = self.m._get_loaded_data('P11_mu2', k)
        
        else:    
            # do the vector part, contributing mu^2 and mu^4 terms
            if not self.m.include_2loop:
                toret = self.m.f**2 * self.m.I31(k)
            else:
                I1 = self.m.Ivvdd_h01(k)
                I2 = self.m.Idvdv_h03(k)
                toret = self.m.f**2 * (I1 + I2)
        
        return toret
        
    def total(self, k):
        return self.vector(k) 
        
class P11_mu4(AngularTerm):
    """
    Implement P11[mu4]
    """
    def vector(self, k):
        return -self.base.mu2.vector(k)
        
    @memoize
    def scalar(self, k):
        # compute the scalar mu^4 contribution
        if self.m.include_2loop:
            I1 = self.m.Ivvdd_h02(k)
            I2 = self.m.Idvdv_h04(k)
            C11_contrib = I1 + I2
        else:
            C11_contrib = self.m.I13(k)

        # the necessary integrals 
        I11 = self.m.I11(k)
        I22 = self.m.I22(k)
        J11 = self.m.J11(k)
        J10 = self.m.J10(k)

        Plin = self.m.normed_power_lin(k)
        part2 = 2*I11 + 4*I22 + 6*k**2 * (J11 + 2*J10)*Plin
        return self.m.f**2 * (Plin + part2 + C11_contrib) - self.vector(k)
        
    @memoize
    def total(self, k):
        # check and return any user-loaded values
        if self.m.P11_mu4_loaded:
            toret = self.m._get_loaded_data('P11_mu4', k)
        else:
            
            # HZPT model
            if self.m.use_P11_model:
                toret = self.m.hzpt.P11(k) - self.m.f**2 * self.m.I31(k)
            # SPT model
            else:
                toret = self.scalar(k) + self.vector(k)
                
        return toret
                    
        
class P11PowerTerm(PowerTerm):
    """
    The full P11 power term
    """
    def __init__(self, model):
        super(P11PowerTerm, self).__init__(model, mu2=P11_mu2, mu4=P11_mu4)
    