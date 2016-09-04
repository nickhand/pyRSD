from .. import GalaxyPowerTerm, ZeroShotNoise

from .PcAsA import PcAsA
from .PcAsB import PcAsB
from .PcBsA import PcBsA
from .PcBsB import PcBsB

class Pcs(GalaxyPowerTerm):
    """
    The cross specturm of central and satellite galaxies
    """
    name = "Pcs"
    
    def __init__(self, model):
        super(Pcs, self).__init__(model, PcAsA, PcAsB, PcBsA, PcBsB)
    
    @property
    def coefficient(self):
        return 2*self.model.fs*(1-self.model.fs)
    
    def __call__(self, k, mu):
        """
        The total central x satellite cross spectrum
        """
        with ZeroShotNoise(self.model):
            toret = super(Pcs, self).__call__(k, mu)
        
        return toret
        