from .. import TwoHaloTerm, OneHaloTerm, DampedGalaxyPowerTerm
   
class PcBs_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcBs`
    """
    name = 'PcBs_2h'
    
    def __init__(self, model):
        super(PcBs_2h, self).__init__(model, 'b1_cB', 'b1_s')
        
class PcBs_1h(OneHaloTerm):
    """
    The 1-halo term for `PcBs`
    """
    name = 'NcBs'  
        
        
class PcBs(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between centrals with satellites
    in the same halo (`cenB`) and satellites
    """
    name = "PcBs"
    
    def __init__(self, model):
        super(PcBs, self).__init__(model, PcBs_2h, PcBs_1h, sigma1='sigma_c', sigma2='sigma_s')
        
    @property
    def coefficient(self):
        return self.model.fcB
        