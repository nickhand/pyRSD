from . import PgalDerivative

class dPgal_dNsBsB(PgalDerivative):
    """
    The partial derivative of `Pgal` with respect to `NsBsB`
    """
    param = 'NsBsB'
    
    @staticmethod
    def eval(m, pars, k, mu):
        
        G = m.evaluate_fog(k, mu, m.sigma_sB)
        return (m.fs*m.fsB*G)**2