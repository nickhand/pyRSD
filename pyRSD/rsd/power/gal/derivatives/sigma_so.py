from . import PgalDerivative
import numpy
from pyRSD.rsd.tools import k_AP, mu_AP

class dPgal_dsigma_so(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``sigma_so``
    """
    param = 'sigma_so'

    @staticmethod
    def eval(m, pars, k, mu):

        if not m.use_so_correction:
            return numpy.zeros(len(k))

        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        G      = m.FOG(kprime, muprime, m.sigma_c)
        G2     = m.FOG(kprime, muprime, m.sigma_so)
        Gprime = m.FOG.derivative_sigma(kprime, muprime, m.sigma_so)

        with m.preserve(use_so_correction=False):

            # Pcc with no FOG kernels
            m.sigma_c = 0.
            Pcc = m.Pgal_cc(k, mu)

            # derivative of the SO correction terms
            term1 = 2*m.f_so*(1-m.f_so) * G * Pcc
            term2 = 2*m.f_so**2 * G2 * Pcc
            term3 = 2*G*m.f_so*m.fcB*m.NcBs / (m.alpha_perp**2 * m.alpha_par)

        toret = (term1 + term2 + term3) * Gprime
        return (1-m.fs)**2 * toret
