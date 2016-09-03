from .. import TwoHaloTerm, DampedGalaxyPowerTerm
from . import SOCorrection

class PcAcA_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcAcA`
    """
    name = 'PcAcA_2h'
    def __init__(self, model):
        super(PcAcA_2h, self).__init__(model, 'b1_cA')
        
    
class PcAcA(DampedGalaxyPowerTerm):
    """
    The auto power spectrum of centrals with no satellites
    in the same halo ('cenA')
    """
    name = "PcAcA"
    
    def __init__(self, model):
        super(PcAcA, self).__init__(model, PcAcA_2h, sigma1='sigma_c')
        
    @property
    def coefficient(self):
        return (1.-self.model.fcB)**2
        
    def __call__(self, k, mu):
        with SOCorrection(self.model):
            return super(PcAcA, self).__call__(k, mu)
            
    def derivative_k(self, k, mu):
        with SOCorrection(self.model):
            return super(PcAcA, self).derivative_k(k, mu)
            
    def derivative_mu(self, k, mu):
        with SOCorrection(self.model):
            return super(PcAcA, self).derivative_mu(k, mu)