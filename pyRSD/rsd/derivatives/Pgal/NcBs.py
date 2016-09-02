from . import PgalDerivative

class dPgal_dNcBs(PgalDerivative):
    """
    The partial derivative of `Pgal` with respect to `NcBs`
    """
    param = 'NcBs'
    
    @staticmethod
    def eval(m, pars, k, mu):

        Gc = m.FOG(k, mu, m.sigma_c)
        Gs = m.FOG(k, mu, m.sigma_s)
        toret = 2*m.fs*(1-m.fs)*m.fcB*Gc*Gs
        
        # additional term from SO correction
        if m.use_so_correction:
            
            G  = m.FOG(k, mu, m.sigma_c)
            G2 = m.FOG(k, mu, m.sigma_so)
            toret += (1-m.fs)**2 * 2*G*G2*m.f_so*m.fcB

        return toret
        