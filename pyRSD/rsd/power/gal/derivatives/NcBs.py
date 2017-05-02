from . import PgalDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPgal_dNcBs(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``NcBs``
    """
    param = 'NcBs'

    @staticmethod
    def eval(m, pars, k, mu):

        # AP shifted
        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        Gc = m.FOG(kprime, muprime, m.sigma_c)
        GsA = m.FOG(kprime, muprime, m.sigma_sA)
        GsB = m.FOG(kprime, muprime, m.sigma_sB)
        toret = 2*m.fs*(1-m.fs)*m.fcB*Gc * ((1-m.fsB)*GsA + m.fsB*GsB)

        # additional term from SO correction
        if m.use_so_correction:

            G1 = m.FOG(kprime, muprime, m.sigma_c)
            G2 = m.FOG(kprime, muprime, m.sigma_so)
            toret += (1-m.fs)**2 * 2*G1*G2*m.f_so*m.fcB

        return toret / (m.alpha_perp**2 * m.alpha_par)
