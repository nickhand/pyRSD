from .. import TwoHaloTerm, DampedGalaxyPowerTerm
from . import SOCorrection

class PcBcB_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcBcB`
    """
    name = 'PcBcB_2h'
    def __init__(self, model):
        super(PcBcB_2h, self).__init__(model, 'b1_cB')
       
    
class PcBcB(DampedGalaxyPowerTerm):
    """
    The auto power spectrum of centrals with
    satellites in the same halo ('cenB')
    """
    name = "PcBcB"
    
    def __init__(self, model):
        super(PcBcB, self).__init__(model, PcBcB_2h, sigma1='sigma_c')
        
    @property
    def coefficient(self):
        return self.model.fcB**2
        
    def __call__(self, k, mu):
        with SOCorrection(self.model):
            return super(PcBcB, self).__call__(k, mu)
            
    def derivative_k(self, k, mu):
        with SOCorrection(self.model):
            return super(PcBcB, self).derivative_k(k, mu)
            
    def derivative_mu(self, k, mu):
        with SOCorrection(self.model):
            return super(PcBcB, self).derivative_mu(k, mu)


