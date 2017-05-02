"""
The derivative of these nuisance parameters with respect to Pgal is zero
"""
from . import PgalDerivative

class dPgal_df1h_sBsB(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``f1h_sBsB``
    """
    param = 'f1h_sBsB'

    @staticmethod
    def eval(m, pars, k, mu):
        return 0.

class dPgal_dNsat_mult(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``Nsat_mult``
    """
    param = 'Nsat_mult'

    @staticmethod
    def eval(m, pars, k, mu):
        return 0.

class dPgal_df1h_cBs(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``f1h_cBs``
    """
    param = 'f1h_cBs'

    @staticmethod
    def eval(m, pars, k, mu):
        return 0.

class dPgal_dlog10_fso(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``log10_fso``
    """
    param = 'log10_fso'

    @staticmethod
    def eval(m, pars, k, mu):
        return 0.
