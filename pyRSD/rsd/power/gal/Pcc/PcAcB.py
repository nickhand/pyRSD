from .. import TwoHaloTerm, DampedGalaxyPowerTerm
from . import SOCorrection
   
class PcAcB_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcAcB`
    """
    name = 'PcAcB_2h'
    def __init__(self, model):
        super(PcAcB_2h, self).__init__(model, 'b1_cA', 'b1_cB')
        
        
class PcAcB(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between centrals with/without
    satellites in the same halo ('cenA' x 'cenB')
    """
    name = "PcAcB"
    
    def __init__(self, model):
        super(PcAcB, self).__init__(model, PcAcB_2h, sigma1='sigma_c')
        
    @property
    def coefficient(self):
        return 2*(1-self.model.fcB)*self.model.fcB
    
    def __call__(self, k, mu):
        with SOCorrection(self.model):
            return super(PcAcB, self).__call__(k, mu)
            
    def derivative_k(self, k, mu):
        with SOCorrection(self.model):
            return super(PcAcB, self).derivative_k(k, mu)
            
    def derivative_mu(self, k, mu):
        with SOCorrection(self.model):
            return super(PcAcB, self).derivative_mu(k, mu)