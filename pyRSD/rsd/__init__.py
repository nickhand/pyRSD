from __future__ import print_function

INTERP_KMIN = 5e-6
INTERP_KMAX = 1.0

# global AP effect lock
import threading
APLock = threading.Lock()

# compute the RSD model version with git string
from ..extern.astropy_helpers.git_helpers import get_git_devstr
from .. import pkg_dir

# compute the version
__version__ = '0.3.1'
_githash = get_git_devstr(sha=True, show_warning=False, path=pkg_dir)[:7]
if _githash:
    __version__ += ".dev." + _githash

# import model classes
from .power.dm     import DarkMatterSpectrum
from .power.biased import BiasedSpectrum, HaloSpectrum
from .power.gal    import GalaxySpectrum
from .power.qso    import QuasarSpectrum

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


class OutdatedModelWarning(UserWarning):
    """
    The warning that will be thrown if the loaded model
    is out-of-date
    """
    pass

def load_model(filename, show_warning=True):
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
    model = numpy.load(filename, encoding='latin1').tolist()

    # check the version
    if show_warning and (not hasattr(model, '__version__') or model.__version__ != __version__):
        import warnings
        msg = "loading an outdated model:\n"
        msg += '\tcurrent model version: %s\n' %(__version__)
        msg += '\tloaded model version: %s\n' %(model.__version__)
        warnings.warn(msg, OutdatedModelWarning)

    return model
