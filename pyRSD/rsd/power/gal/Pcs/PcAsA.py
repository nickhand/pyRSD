from .. import TwoHaloTerm, DampedGalaxyPowerTerm
   
class PcAsA_2h(TwoHaloTerm):
    """
    The 2-halo term for `PcAsA`
    """
    name = 'PcAsA_2h'
    def __init__(self, model):
        super(PcAsA_2h, self).__init__(model, 'b1_cA', 'b1_sA')
        
        
class PcAsA(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between centrals with no satellites
    in the same halo (`cenA`) and satellites with a single 
    satellite (`satA`)
    """
    name = "PcAsA"
    
    def __init__(self, model):
        super(PcAsA, self).__init__(model, PcAsA_2h, sigma1='sigma_c', sigma2='sigma_sA')
        
    @property
    def coefficient(self):
        return (1.-self.model.fcB) * (1.-self.model.fsB)
