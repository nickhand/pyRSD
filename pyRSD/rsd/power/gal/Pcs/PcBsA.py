from .. import TwoHaloTerm, OneHaloTerm, DampedGalaxyPowerTerm
   
class PcBsA_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcBsA`
    """
    name = 'PcBsA_2h'
    
    def __init__(self, model):
        super(PcBsA_2h, self).__init__(model, 'b1_cB', 'b1_sA')
            
class PcBsA_1h(OneHaloTerm):
    """
    The 1-halo term for `PcBsA`
    """
    name = 'NcBs'  
        
        
class PcBsA(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between centrals with satellites
    in the same halo (`cenB`) and sats with only 
    1 satellite (`satA`)
    """
    name = "PcBsA"
    
    def __init__(self, model):
        super(PcBsA, self).__init__(model, PcBsA_2h, PcBsA_1h, sigma1='sigma_c', sigma2='sigma_sA')
        
    @property
    def coefficient(self):
        return self.model.fcB * (1.-self.model.fsB)
        