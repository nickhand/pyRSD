from . import PgalDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPgal_dsigma_c(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``sigma_c``
    """
    param = 'sigma_c'

    @staticmethod
    def eval(m, pars, k, mu):

        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        G = m.FOG(kprime, muprime, m.sigma_c)
        Gprime = m.FOG.derivative_sigma(kprime, muprime, m.sigma_c)

        with m.preserve():
            m.sigma_c = 0

            if not m.use_so_correction:
                term1 = 2*G*Gprime * m.Pgal_cc(k, mu)
            else:
                # turn off SO correction
                m.use_so_correction = False
                Pcc = m.Pgal_cc(k, mu)

                # derivative of the SO correction terms
                G2    = m.FOG(kprime, muprime, m.sigma_so)
                term1_a = 2*G* (1-m.f_so)**2 * Pcc
                term1_b = 2*m.f_so*(1-m.f_so) * G2 * Pcc
                term1_c = 2*G2*m.f_so*m.fcB*m.NcBs / (m.alpha_perp**2 * m.alpha_par)
                term1 = (term1_a + term1_b + term1_c) * Gprime

            term2 = Gprime * m.Pgal_cs(k, mu)

        return (1-m.fs)**2 * term1 + 2*m.fs*(1-m.fs) * term2
