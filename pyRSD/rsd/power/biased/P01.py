from .. import AngularTerm, PowerTerm, memoize

def P01_mu2(self, k, b2_01_func):
    """
    The correlation of the halo density and halo momentum fields, which 
    contributes mu^2 terms to the power expansion.
    """ 
    # the bias values to use
    b1, b1_bar = self._ib1, self._ib1_bar
    b2_01, b2_01_bar = b2_01_func(b1), b2_01_func(b1_bar)
    
    # the relevant PT integrals
    K10  = self.K10(k)
    K10s = self.K10s(k)
    K11  = self.K11(k)
    K11s = self.K11s(k)

    term1 = (b1*b1_bar) * self.P01.mu2(k)
    term2 = -self.Pdv(k) * (b1*(1. - b1_bar) + b1_bar*(1. - b1))
    term3 = self.f*((b2_01 + b2_01_bar)*K10 + (self.bs + self.bs_bar)*K10s )
    term4 = self.f*((b1_bar*b2_01 + b1*b2_01_bar)*K11 + (b1_bar*self.bs + b1*self.bs_bar)*K11s)
    return term1 + term2 + term3 + term4
    

class P01AngularTerm(AngularTerm):
    """
    Implement P01[mu2]
    """
    @memoize
    def total(self, k):
        return P01_mu2(self.m, k, self.m.b2_01_a)


class P01PowerTerm(PowerTerm):
    """
    The full P01 power term
    """
    def __init__(self, model):
        super(P01PowerTerm, self).__init__(model, mu2=P01AngularTerm)
        
    
