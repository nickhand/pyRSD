from .. import TwoHaloTerm, OneHaloTerm, DampedGalaxyPowerTerm
   
class PcBsB_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcBsB`
    """
    name = 'PcBsB_2h'
    
    def __init__(self, model):
        super(PcBsB_2h, self).__init__(model, 'b1_cB', 'b1_sB')
            
class PcBsB_1h(OneHaloTerm):
    """
    The 1-halo term for `PcBsB`
    """
    name = 'NcBs'  
        
        
class PcBsB(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between centrals with satellites
    in the same halo (`cenB`) and >1 satellites (`satB`)
    """
    name = "PcBsB"
    
    def __init__(self, model):
        super(PcBsB, self).__init__(model, PcBsB_2h, PcBsB_1h, sigma1='sigma_c', sigma2='sigma_sB')
        
    @property
    def coefficient(self):
        return self.model.fcB * self.model.fsB
        