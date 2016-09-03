from .. import AngularTerm, PowerTerm, memoize

class P00_mu0(AngularTerm):
    """
    Implement P00[mu0]
    """
    @memoize
    def total(self, k):
        
        # check and return any user-loaded values
        if self.m.P00_mu0_loaded:
            toret = self.m._get_loaded_data('P00_mu0', k)
        else:
                
            # use the DM model
            if self.m.use_P00_model:
                toret = self.m.hzpt.P00(k)
            # use pure PT
            else:        
                P11 = self.m.normed_power_lin(k)
                P22 = 2*self.m.I00(k)
                P13 = 6*k**2*self.m.J00(k)*P11
                toret = P11 + P22 + P13
            
        return toret
        
class P00PowerTerm(PowerTerm):
    """
    The full P00 power term
    """
    def __init__(self, model):
        super(P00PowerTerm, self).__init__(model, mu0=P00_mu0)
    