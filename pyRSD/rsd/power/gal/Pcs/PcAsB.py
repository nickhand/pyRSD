from .. import TwoHaloTerm, DampedGalaxyPowerTerm
   
class PcAsB_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcAsB`
    """
    name = 'PcAsB_2h'
    def __init__(self, model):
        super(PcAsB_2h, self).__init__(model, 'b1_cA', 'b1_sB')
        
        
class PcAsB(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between centrals with no satellites
    in the same halo (`cenA`) and satellites with >1  
    satellite (`satB`)
    """
    name = "PcAsB"
    
    def __init__(self, model):
        super(PcAsB, self).__init__(model, PcAsB_2h, sigma1='sigma_c', sigma2='sigma_sB')
        
    @property
    def coefficient(self):
        return (1.-self.model.fcB) * self.model.fsB
