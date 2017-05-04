from . import PqsoDerivative
from pyRSD.rsd.tools import k_AP, mu_AP

class dPqso_df_nl(PqsoDerivative):
    """
    The partial derivative of :func:`QuasarSpectrum.power` with respect to
    ``f_nl``
    """
    param = 'f_nl'

    @staticmethod
    def eval(m, pars, k, mu):

        kprime  = k_AP(k, mu, m.alpha_perp, m.alpha_par)
        muprime = mu_AP(mu, m.alpha_perp, m.alpha_par)

        # finger-of-god
        G = m.FOG(kprime, muprime, m.sigma_fog)

        # derivative of scale-dependent bias term
        ddb_dfnl = 2*(m.b1-m.p)*m.delta_crit/m.alpha_png(kprime)

        # do the volume rescaling
        rescaling = (m.alpha_drag**3) / (m.alpha_perp**2 * m.alpha_par)

        # the final derivative
        btot = m.btot(kprime)
        return rescaling * G**2 * (2 * m.P_mu0(kprime) + m.P_mu2(kprime)*muprime**2) / btot * ddb_dfnl
