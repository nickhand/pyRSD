"""
pyRSD

``pyRSD`` is a collection of algorithms to compute the redshift space matter
power spectra using perturbation theory and the redshift space distortion (RSD)
model based on a distribution function velocity moments approach

for all features of ``pyRSD``, you need to import one of the
following subpackages:

Subpackages
-----------
data
    Simulation data.
rsd
    RSD power spectra.
pygcl
    Python bindings for a C++ "General Cosmology Library"
"""

# save the absolute path of the package and data directories
import os.path as _osp
import sys
import os

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# every module uses numpy
import numpy

try:
    from pyRSD import pygcl
except Exception as msg:
    if on_rtd:
        pygcl = None
    else:
        import traceback
        tb = traceback.format_exc()
        raise ImportError("Cannot use package without pygcl\n%s" %tb)

def get_data_files():
    """
    Returns the path of data files, which are installed to the package directory.
    """

    import os

    path = os.path.dirname(__file__)
    path = os.path.join(path, 'data', 'class')
    r = dict(
        Alpha_inf_hyrec_file = os.path.join(path, 'hyrec', 'Alpha_inf.dat'),
        R_inf_hyrec_file = os.path.join(path, 'hyrec', 'R_inf.dat'),
        two_photon_tables_hyrec_file = os.path.join(path, 'hyrec', 'two_photon_tables.dat'),
        sBBN_file = os.path.join(path, 'bbn', 'sBBN.dat'),
    )
    return r

def _init():
    r = get_data_files()

    # setting static variables with swig is tricky.
    # see http://www.swig.org/Doc3.0/SWIGDocumentation.html#Python_nn20

    from .gcl import cvar

    cvar.ClassEngine_Alpha_inf_hyrec_file = r['Alpha_inf_hyrec_file']
    cvar.ClassEngine_R_inf_hyrec_file = r['R_inf_hyrec_file']
    cvar.ClassEngine_two_photon_tables_hyrec_file = r['two_photon_tables_hyrec_file']
    cvar.ClassEngine_sBBN_file = r['sBBN_file']

if pygcl is not None:
    _init(); del _init

from .version import __version__
