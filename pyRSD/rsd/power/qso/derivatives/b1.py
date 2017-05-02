from . import PqsoDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPqso_db1(PqsoDerivative):
    """
    The partial derivative of :func:`QuasarSpectrum.power` with respect to
    ``b1``
    """
    param = 'b1'

    @staticmethod
    def eval(m, pars, k, mu):

        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        G = m.FOG(kprime, muprime, m.sigma_fog)
        return G**2 * (2 * m.P_mu0(kprime) + m.P_mu2(kprime)*muprime**2) / m.b1
