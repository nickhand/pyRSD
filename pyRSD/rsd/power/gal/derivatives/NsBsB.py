from . import PgalDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPgal_dNsBsB(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``NsBsB``
    """
    param = 'NsBsB'

    @staticmethod
    def eval(m, pars, k, mu):

        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)
        G = m.FOG(kprime, muprime, m.sigma_sB)

        return (m.fs*m.fsB*G)**2 / (m.alpha_perp**2 * m.alpha_par)
