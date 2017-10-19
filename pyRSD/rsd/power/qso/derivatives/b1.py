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

        # derivative of scale-dependent bias term
        dbtot_db1 = 1 + 2*m.f_nl*m.delta_crit/m.alpha_png(kprime)

        # finger-of-god
        G = m.FOG(kprime, muprime, m.sigma_fog)

        # the final derivative
        btot = m.btot(kprime)
        rescaling = (m.alpha_drag**3) / (m.alpha_perp**2 * m.alpha_par)
        return rescaling * G**2 * (2*m.P_mu0(kprime)  + muprime**2 * m.P_mu2(kprime)) * dbtot_db1 / btot
