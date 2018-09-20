from . import PqsoDerivative
from pyRSD.rsd.tools import k_AP, mu_AP
import numpy as np


class dPqso_dsigma_fog(PqsoDerivative):
    """
    The partial derivative of :func:`QuasarSpectrum.power` with respect to
    ``sigma_fog``
    """
    param = 'sigma_fog'

    @staticmethod
    def eval(m, pars, k, mu):

        kprime = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        G = m.FOG(kprime, muprime, m.sigma_fog)
        if hasattr(G, 'values'):
            G = G.values
        Gprime = m.FOG.derivative_sigma(kprime, muprime, m.sigma_fog)
        toret = np.zeros_like(G)

        valid = np.nonzero(G)
        toret[valid] = 2*Gprime[valid] * m.power(k, mu)[valid] / G[valid]
        return toret
