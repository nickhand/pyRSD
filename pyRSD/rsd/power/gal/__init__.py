from .core import OneHaloTerm, TwoHaloTerm
from .core import GalaxyPowerTerm, DampedGalaxyPowerTerm

class ZeroShotNoise(object):
    """
    Class to manage the handling of `N` when computing component spectra
    """
    def __init__(self, model):
        self.model = model
        self._N = self.model.N
        
    def __enter__(self):
        self.model.N = 0.

    def __exit__(self, *args):
        self.model.N = self._N
        

from .Pcc import Pcc
from .Pcs import Pcs
from .Pss import Pss

        
class Pgal(GalaxyPowerTerm):
    """
    The auto specturm of cental + satellite galaxies
    """
    name = "Pgal"
    
    def __init__(self, model):
        super(Pgal, self).__init__(model, Pcc, Pcs, Pss)
        
    def __call__(self, k, mu):
        """
        The total galaxy auto spectrum
        """
        with ZeroShotNoise(self.model):
            toret = super(Pgal, self).__call__(k, mu)
        
        return toret
        
from .power_gal import GalaxySpectrum