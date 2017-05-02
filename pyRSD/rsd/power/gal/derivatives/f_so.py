from . import PgalDerivative
from pyRSD.rsd.tools import k_AP, mu_AP
import numpy

class dPgal_df_so(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``f_so``
    """
    param = 'f_so'

    @staticmethod
    def eval(m, pars, k, mu):

        if not m.use_so_correction:
            return numpy.zeros(len(k))

        # sum the individual components of Pcc
        PcAcA = (1.-m.fcB)**2 * m.Pgal_cAcA(k, mu)
        PcAcB = 2*m.fcB*(1-m.fcB)*m.Pgal_cAcB(k, mu)
        PcBcB = m.fcB**2 * m.Pgal_cBcB(k, mu)
        Pcc = PcAcA + PcAcB + PcBcB

        # FOG
        kprime = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)
        G1 = m.FOG(kprime, muprime, m.sigma_c)
        G2 = m.FOG(kprime, muprime, m.sigma_so)

        # the SO correction
        term1 = -2 * (1. - m.f_so) * G1**2 * Pcc
        term2 = 2 * (1 - 2*m.f_so) * G1*G2 * Pcc
        term3 = 2 * m.f_so * G2**2 * Pcc
        term4 = 2*G1*G2*m.fcB*m.NcBs / m.alpha_perp**2 / m.alpha_par

        return (1-m.fs)**2 * (term1 + term2 + term3 + term4)
