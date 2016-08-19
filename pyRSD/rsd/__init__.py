INTERP_KMIN = 5e-6
INTERP_KMAX = 1.0

from .power_dm import DarkMatterSpectrum
from .power_biased import BiasedSpectrum
from .power_halo import HaloSpectrum
from .power_gal import GalaxySpectrum
from .grid_transfer import PkmuTransfer, PolesTransfer, PkmuGrid
from .power_extrapolator import ExtrapolatedPowerSpectrum
from .correlation import SmoothedXiMultipoles

__version__ = DarkMatterSpectrum.__version__
def print_version():
    import time
    print("RSD model version %s (%s)" %(__version__, time.ctime()))

__all__ = ['power_dm', 'power_biased', 'power_halo', 'power_gal', 'grid_transfer', 'correlation']