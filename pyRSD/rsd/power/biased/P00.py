from .. import AngularTerm, PowerTerm, memoize

def Phh_mu0(self, k, b2_00_func):
    """
    The 1-loop SPT halo density field auto-correlation
    """ 
    # the bias values to use
    b1, b1_bar = self._ib1, self._ib1_bar
    b2_00, b2_00_bar = b2_00_func(b1), b2_00_func(b1_bar)
    
    term1 = b1*b1_bar * self.P00.mu0(k)
    term2 = (b1*b2_00_bar + b1_bar*b2_00)*self.K00(k)
    term3 = (self.bs*b2_00_bar + self.bs_bar*b2_00)*self.K00s(k)
    return term1 + term2 + term3

class P00_mu0(AngularTerm):
    """
    Implement P00[mu0]
    """
    @memoize
    def total(self, k):
        return self.m.P00_ss_no_stoch.mu0(k) + self.m.stochasticity(k)

        
class P00_no_stoch_mu0(AngularTerm):
    """
    Implement P00[mu0] without stochasticity
    """
    @memoize
    def total(self, k):
        
        P00 = self.m.P00.mu0(k)
        b1_k     = self.m.Phm(k) / P00
        b1_bar_k = self.m.Phm_bar(k) / P00
        
        return b1_k*b1_bar_k * P00

class P00PowerTerm(PowerTerm):
    """
    The full P00 power term
    """
    def __init__(self, model):
        super(P00PowerTerm, self).__init__(model, mu0=P00_mu0)
        
class NoStochP00PowerTerm(PowerTerm):
    """
    The P00 power term without stochasticity
    """
    def __init__(self, model):
        super(NoStochP00PowerTerm, self).__init__(model, mu0=P00_no_stoch_mu0)
    
