from . import PgalDerivative
from .fog_kernels import get_fog_derivative
import numpy

class dPgal_dsigma_so(PgalDerivative):
    """
    The partial derivative of `Pgal` with respect to `sigma_so`
    """
    param = 'sigma_so'
    
    @staticmethod
    def eval(m, pars, k, mu):
        
        if not m.use_so_correction:
            return numpy.zeros(len(k))
        
        G      = m.evaluate_fog(k, mu, m.sigma_c)
        G2     = m.evaluate_fog(k, mu, m.sigma_so)        
        Gprime = k*mu * get_fog_derivative(m.fog_model, k*mu*m.sigma_so)

        with m.preserve(use_so_correction=False):
            
            # Pcc with no FOG kernels
            m.sigma_c = 0.
            Pcc = m.Pgal_cc(k, mu)

            # derivative of the SO correction terms    
            term1 = 2*m.f_so*(1-m.f_so) * G * Pcc
            term2 = 2*m.f_so**2 * G2 * Pcc
            term3 = 2*G*m.f_so*m.fcB*m.NcBs
            
        toret = (term1 + term2 + term3) * Gprime    
        return (1-m.fs)**2 * toret
            
