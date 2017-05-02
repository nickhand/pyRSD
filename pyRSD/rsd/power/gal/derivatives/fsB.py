from . import PgalDerivative

class dPgal_dfsB(PgalDerivative):
    """
    The partial derivative of :func:`GalaxySpectrum.power` with respect to
    ``fsB``
    """
    param = 'fsB'

    @staticmethod
    def eval(m, pars, k, mu):

        toret = 2 * ( -(1-m.fsB)*m.Pgal_sAsA(k,mu) + (1-2*m.fsB)*m.Pgal_sAsB(k,mu) + m.fsB*m.Pgal_sBsB(k,mu))
        return m.fs**2 * toret
