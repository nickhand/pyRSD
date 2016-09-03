from .. import AngularTerm, PowerTerm, memoize
  
class P11_mu2(AngularTerm):
    """
    Implement P11[mu2]
    """
    @memoize
    def total(self, k):
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        
        # this is C11 at 2-loop order
        I1 = self.m.Ivvdd_h01(k)
        I2 = self.m.Idvdv_h03(k)
        return (b1*b1_bar)*self.m.f**2 * (I1 + I2)
        
class P11_mu4(AngularTerm):
    """
    Implement P11[mu4]
    """
    @memoize
    def total(self, k):
        b1, b1_bar = self.m._ib1, self.m._ib1_bar
        Plin = self.m.normed_power_lin(k)
        
        # get the integral attributes
        I22 = self.m.I22(k)
        J10 = self.m.J10(k)
        
        # first term is mu^4 part of P11
        term1_mu4 = 0.5*(b1 + b1_bar)*self.m.P11.mu4(k)
        
        # second term is B11 coming from P11
        term2_mu4 = -0.5*((b1 - 1) + (b1_bar-1)) * self.m.Pvv(k)
        
        # third term is mu^4 part of C11 (at 2-loop)
        I1 = self.m.Ivvdd_h02(k)
        I2 = self.m.Idvdv_h04(k)
        term3_mu4 = self.m.f**2 * (I1 + I2) * (b1*b1_bar - 0.5*(b1 + b1_bar))

        return term1_mu4 + term2_mu4 + term3_mu4


class P11PowerTerm(PowerTerm):
    """
    The full P11 power term
    """
    def __init__(self, model):
        super(P11PowerTerm, self).__init__(model, mu2=P11_mu2, mu4=P11_mu4)
        
    
