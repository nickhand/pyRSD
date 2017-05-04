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

        # finger of god
        G = m.FOG(kprime, muprime, m.sigma_fog)

        # derivative of scale-dependent bias term
        dbtot_db1 = 1 + 2*m.f_nl*m.delta_crit/m.alpha_png(kprime)

        # do the volume rescaling
        rescaling = (m.alpha_drag**3) / (m.alpha_perp**2 * m.alpha_par)

        # the final derivative
        btot = m.btot(kprime)
        Plin = m.normed_power_lin(kprime)
        return rescaling * G**2 * (2*btot*Plin + 2*m.f*muprime**2*Plin)*dbtot_db1
