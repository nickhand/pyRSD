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

# try to import pygcl; hopefully you succeed 
sys.path.insert(1, "%s/gcl/python" %pkg_dir)
try:
    import pygcl
except Exception as msg:
    raise ImportError("Cannot use package without pygcl; original message: %s" %msg)
    
# every module uses numpy
import numpy
import plotify