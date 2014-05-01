"""pyRSD

``pyRSD`` is a collection of algorithms to compute the redshift space matter 
power spectra using perturbation theory and the redshift space distortion (RSD) 
model based on a distribution function velocity moments approach

for all features of ``pyRSD``, you need to import one of the
following subpackages:

Subpackages
-----------
cosmology
    Cosmological calculations.
data
    Simulation data.
rsd
    RSD power spectra.
"""
import os.path as _osp

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')