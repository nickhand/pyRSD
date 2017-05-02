from . import PgalDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPgal_dfcB(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``fcB``
    """
    param = 'fcB'

    @staticmethod
    def eval(m, pars, k, mu):

        term1 = 2*( -(1-m.fcB)*m.Pgal_cAcA(k,mu) + (1-2*m.fcB)*m.Pgal_cAcB(k,mu) + m.fcB*m.Pgal_cBcB(k,mu))
        term2 = (1-m.fsB)*(-m.Pgal_cAsA(k,mu) + m.Pgal_cBsA(k,mu)) + m.fsB*(-m.Pgal_cAsB(k,mu) + m.Pgal_cBsB(k,mu))

        # AP shifted
        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        # additional term from SO correction
        if m.use_so_correction:

            G1 = m.FOG(kprime, muprime, m.sigma_c)
            G2 = m.FOG(kprime, muprime, m.sigma_so)
            term1 *= (((1 - m.f_so)*G1)**2 + 2*m.f_so*(1-m.f_so)*G1*G2 + (m.f_so*G2)**2)
            term1 += 2*G1*G2*m.f_so*m.NcBs / (m.alpha_perp**2 * m.alpha_par)

        return (1-m.fs)**2  * term1 + 2*m.fs*(1-m.fs) * term2
