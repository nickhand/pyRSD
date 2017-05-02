from . import PqsoDerivative

class dPqso_dsigma8_z(PqsoDerivative):
    """
    The partial derivative of :func:`QuasarSpectrum.power` with respect to
    ``b1``
    """
    param = 'sigma8_z'

    @staticmethod
    def eval(m, pars, k, mu):

        return 2 * m.power(k,mu) / m.sigma8_z
