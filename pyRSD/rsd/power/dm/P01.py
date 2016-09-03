from .. import AngularTerm, PowerTerm, memoize

class P01_mu2(AngularTerm):
    """
    Implement P01[mu2]
    """
    @memoize
    def total(self, k):
                    
        # check and return any user-loaded values
        if self.m.P01_mu2_loaded:
            toret = self.m._get_loaded_data('P01_mu2', k)
        else:
        
            # HZPT P01 model
            if self.m.use_P01_model:
                toret = self.m.hzpt.P01(k)
            # pure SPT
            else:                
                # the necessary integrals 
                I00 = self.m.I00(k)
                J00 = self.m.J00(k)
    
                Plin = self.m.normed_power_lin(k)
                toret = 2*self.m.f*(Plin + 4.*(I00 + 3*k**2*J00*Plin))
        
        return toret
        
class P01PowerTerm(PowerTerm):
    """
    The full P01 power term
    """
    def __init__(self, model):
        super(P01PowerTerm, self).__init__(model, mu2=P01_mu2)
    