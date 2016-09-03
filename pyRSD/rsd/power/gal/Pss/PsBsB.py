from .. import TwoHaloTerm, OneHaloTerm, DampedGalaxyPowerTerm

class PsBsB_2h(TwoHaloTerm):
    """
    The 2-halo term for `PsBsB`
    """
    name = 'PsBsB_2h'
    def __init__(self, model):
        super(PsBsB_2h, self).__init__(model, 'b1_sB')

class PsBsB_1h(OneHaloTerm):
    """
    The 1-halo term for `PsBsB`
    """
    name = 'NsBsB'        
    
class PsBsB(DampedGalaxyPowerTerm):
    """
    The auto power spectrum of satellites with other
    satellites in the same halo ('satB')
    """
    name = "PsBsB"
    
    def __init__(self, model):
        super(PsBsB, self).__init__(model, PsBsB_2h, PsBsB_1h, sigma1='sigma_sB')
        
    @property
    def coefficient(self):
        return self.model.fsB**2
        


