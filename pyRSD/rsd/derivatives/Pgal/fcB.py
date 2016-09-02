from . import PgalDerivative

class dPgal_dfcB(PgalDerivative):
    """
    The partial derivative of `Pgal` with respect to `fcB`
    """
    param = 'fcB'
    
    @staticmethod
    def eval(m, pars, k, mu):
        
        term1 = 2*( -(1-m.fcB)*m.Pgal_cAcA(k,mu) + (1-2*m.fcB)*m.Pgal_cAcB(k,mu) + m.fcB*m.Pgal_cBcB(k,mu))
        term2 = (-m.Pgal_cAs(k,mu) + m.Pgal_cBs(k,mu))
        
        # additional term from SO correction
        if m.use_so_correction:
            G  = m.FOG(k, mu, m.sigma_c)
            G2 = m.FOG(k, mu, m.sigma_so)            
            term1 *= (((1 - m.f_so)*G)**2 + 2*m.f_so*(1-m.f_so)*G*G2 + (m.f_so*G2)**2)
            term1 += 2*G*G2*m.f_so*m.NcBs
            
        return (1-m.fs)**2  * term1 + 2*m.fs*(1-m.fs) * term2
        


    
