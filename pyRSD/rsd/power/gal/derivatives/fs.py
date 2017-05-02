from . import PgalDerivative

class dPgal_dfs(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to ``fs``
    """
    param = 'fs'

    @staticmethod
    def eval(m, pars, k, mu):

        term1 = -2 * (1. - m.fs) * m.Pgal_cc(k, mu)
        term2 = 2 * (1 - 2*m.fs) * m.Pgal_cs(k, mu)
        term3 = 2 * m.fs * m.Pgal_ss(k, mu)
        return term1 + term2 + term3
