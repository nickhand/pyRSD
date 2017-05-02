from . import PgalDerivative
from pyRSD.rsd import APLock

class dPgal_dalpha_perp(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``alpha_perp``
    """
    param = 'alpha_perp'

    @staticmethod
    def eval(m, pars, k, mu):

        # some setup
        aperp = m.alpha_perp
        apar  = m.alpha_par
        F     = apar/aperp
        N     = (1. + mu**2 * (1/F**2 - 1.))

        # derivative of the volume factor
        term1 = -2. * m.power(k, mu) / aperp

        # derivatives wrt to kprime, muprime
        dPgal_dkprime = m.derivative_k(k, mu)
        dPgal_dmuprime = m.derivative_mu(k, mu)

        # derivative of remappings wrt alpha_perp
        dkprime_da = -k*(1. - mu**2) / aperp**2 / N**0.5
        dmuprime_da = mu * (1. - mu**2) / (apar * N**1.5)

        return term1 + dPgal_dkprime * dkprime_da + dPgal_dmuprime * dmuprime_da
