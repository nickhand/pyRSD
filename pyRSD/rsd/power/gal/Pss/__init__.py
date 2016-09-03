from .. import GalaxyPowerTerm, ZeroShotNoise

from .PsAsA import PsAsA
from .PsAsB import PsAsB
from .PsBsB import PsBsB

        
class Pss(GalaxyPowerTerm):
    """
    The auto specturm of satellite galaxies
    """
    name = "Pss"
    
    def __init__(self, model):
        super(Pss, self).__init__(model, PsAsA, PsAsB, PsBsB)
    
    @property
    def coefficient(self):
        return self.model.fs**2 
    
    def __call__(self, k, mu, flatten=False):
        """
        The total satellites auto spectrum
        """
        with ZeroShotNoise(self.model):
            toret = super(Pss, self).__call__(k, mu)
        
        return toret