from . import PqsoDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPqso_df(PqsoDerivative):
    """
    The partial derivative of :func:`QuasarSpectrum.power` with respect to ``f``
    """
    param = 'f'

    @staticmethod
    def eval(m, pars, k, mu):

        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        # finger of god
        G = m.FOG(kprime, muprime, m.sigma_fog)

        # do the volume rescaling
        rescaling = (m.alpha_drag**3) / (m.alpha_perp**2 * m.alpha_par)

        mu2 = muprime**2
        return rescaling * G**2 * mu2 * (m.P_mu2(kprime) + 2 * m.P_mu4(kprime)*mu2) / m.f
