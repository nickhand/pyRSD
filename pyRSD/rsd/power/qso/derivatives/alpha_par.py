from . import PqsoDerivative

class dPqso_dalpha_par(PqsoDerivative):
    """
    The partial derivative of :func:`QuasarSpectrum.power` with respect to
    ``alpha_par``
    """
    param = 'alpha_par'

    @staticmethod
    def eval(m, pars, k, mu):

        # some setup
        aperp = m.alpha_perp
        apar  = m.alpha_par
        F     = apar/aperp
        N     = (1. + mu**2 * (1/F**2 - 1.))

        # derivative of the volume factor
        term1 = - m.power(k, mu) / apar

        # derivatives wrt to kprime, muprime
        dPkmu_dkprime = m.derivative_k(k, mu)
        dPkmu_dmuprime = m.derivative_mu(k, mu)

        # derivative of AP remapping
        dkprime_da  = -k*mu**2 / (F*apar**2) / N**0.5
        dmuprime_da = -mu*(1. - mu**2) / (F * apar * N**1.5)

        return term1 + dPkmu_dkprime * dkprime_da + dPkmu_dmuprime * dmuprime_da
