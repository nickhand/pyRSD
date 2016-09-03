from .. import TwoHaloTerm, DampedGalaxyPowerTerm
   
class PsAsB_2h(TwoHaloTerm):
    """
    The 2-halo term for `PsAsB`
    """
    name = 'PsAsB_2h'
    def __init__(self, model):
        super(PsAsB_2h, self).__init__(model, 'b1_sA', 'b1_sB')
        
        
class PsAsB(DampedGalaxyPowerTerm):
    """
    The cross power spectrum between satellites with/without other
    satellites in the same halo ('satA' x 'satB')
    """
    name = "PsAsB"
    
    def __init__(self, model):
        super(PsAsB, self).__init__(model, PsAsB_2h, sigma1='sigma_sA', sigma2='sigma_sB')
        
    @property
    def coefficient(self):
        return 2*(1-self.model.fsB)*self.model.fsB
        