from . import PgalDerivative
import numpy

class dPgal_df_so(PgalDerivative):
    """
    The partial derivative of `Pgal` with respect to `f_so`
    """
    param = 'f_so'
    
    @staticmethod
    def eval(m, pars, k, mu):
        
        if not m.use_so_correction:
            return numpy.zeros(len(k))
            
        # sum the individual components of Pcc                    
        PcAcA = (1.-m.fcB)**2 * m.Pgal_cAcA(k, mu)
        PcAcB = 2*m.fcB*(1-m.fcB)*m.Pgal_cAcB(k, mu)
        PcBcB = m.fcB**2 * m.Pgal_cBcB(k, mu)
        Pcc = PcAcA + PcAcB + PcBcB
        
        # the SO correction
        G = m.evaluate_fog(k, mu, m.sigma_c)
        G2 = m.evaluate_fog(k, mu, m.sigma_so)
        term1 = -2 * (1. - m.f_so) * G**2 * Pcc
        term2 = 2 * (1 - 2*m.f_so) * G*G2 * Pcc
        term3 = 2 * m.f_so * G2**2 * Pcc
        term4 = 2*G*G2*m.fcB*m.NcBs

        return (1-m.fs)**2 * (term1 + term2 + term3 + term4)