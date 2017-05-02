from . import PqsoDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPqso_dsigma_fog(PqsoDerivative):
    """
    The partial derivative of :func:`QuasarSpectrum.power` with respect to
    ``sigma_fog``
    """
    param = 'sigma_fog'

    @staticmethod
    def eval(m, pars, k, mu):

        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        G = m.FOG(kprime, muprime, m.sigma_fog)
        Gprime = m.FOG.derivative_sigma(kprime, muprime, m.sigma_fog)
        return 2*Gprime * m.power(k, mu) / G
