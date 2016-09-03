from .. import TwoHaloTerm, DampedGalaxyPowerTerm
   
class PcAs_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcAs`
    """
    name = 'PcAs_2h'
    def __init__(self, model):
        super(PcAs_2h, self).__init__(model, 'b1_cA', 'b1_s')
        
        
class PcAs(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between centrals with no satellites
    in the same halo (`cenA`) and satellites
    """
    name = "PcAs"
    
    def __init__(self, model):
        super(PcAs, self).__init__(model, PcAs_2h, sigma1='sigma_c', sigma2='sigma_s')
        
    @property
    def coefficient(self):
        return 1.-self.model.fcB
