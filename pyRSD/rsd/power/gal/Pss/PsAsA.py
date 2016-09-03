from .. import TwoHaloTerm, DampedGalaxyPowerTerm

class PsAsA_2h(TwoHaloTerm):
    """
    The 2-halo term for `PsAsA`
    """
    name = 'PsAsA_2h'
    def __init__(self, model):
        super(PsAsA_2h, self).__init__(model, 'b1_sA')
        
        
class PsAsA(DampedGalaxyPowerTerm):
    """
    The auto power spectrum of satellites with no other
    satellites in the same halo ('satA')
    """
    name = "PsAsA"
    
    def __init__(self, model):
        super(PsAsA, self).__init__(model, PsAsA_2h, sigma1='sigma_sA')
        
    @property
    def coefficient(self):
        return (1.-self.model.fsB)**2