INTERP_KMIN = 5e-6
INTERP_KMAX = 1.0

# compute the RSD model version with git string
from astropy_helpers.git_helpers import get_git_devstr
from .. import pkg_dir
__version__ = '0.2.0' + '-' + get_git_devstr(sha=True, path=pkg_dir)[:7]

# import model classes
from .power_dm import DarkMatterSpectrum
from .power_biased import BiasedSpectrum
from .power_halo import HaloSpectrum
from .power_gal import GalaxySpectrum
from .grid_transfer import PkmuTransfer, PolesTransfer, PkmuGrid
from .power_extrapolator import ExtrapolatedPowerSpectrum
from .correlation import SmoothedXiMultipoles


def print_version():
    """
    Print out the RSD model version
    """
    import time
    print("RSD model version %s (%s)" %(__version__, time.ctime()))

__all__ = ['power_dm', 'power_biased', 'power_halo', 'power_gal', 'grid_transfer', 'correlation']


def load_model(filename):
    """
    Load a model from a npy
    """
    from .. import os, numpy
    
    # check the filename extension
    _, ext = os.path.splitext(filename)
    desired_ext = os.path.extsep + 'npy'
    if ext != desired_ext:
        raise ValueError("file name should end in %s" %desired_ext)
        
    # load
    model = numpy.load(filename).tolist()
    
    # check the version
    if hasattr(model, '__version__') and model.__version__ != __version__:
        import warnings
        msg = "loading an outdated model:\n"
        msg += '\tcurrent model version: %s\n' %(__version__)
        msg += '\tloaded model version: %s\n' %(model.__version__)
        warnings.warn(msg)
        
    return model