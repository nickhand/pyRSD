from . import PgalDerivative
from pyRSD.rsd import APLock

class dPgal_dalpha_perp(PgalDerivative):
    """
    The partial derivative of `Pgal` with respect to `alpha_perp`
    """
    param = 'alpha_perp'
    
    @staticmethod
    def eval(m, pars, k, mu, epsilon=1e-3):
        
        aperp = m.alpha_perp
        apar  = m.alpha_par
        
        # derivative of the volume factor
        term1 = -2. * m.Pgal(k, mu) / aperp
        
        # central finite difference derivative wrt to k,mu
        with APLock:
            gradk = (m.Pgal(k+epsilon, mu) - m.Pgal(k-epsilon, mu)) / (2.*epsilon)
            gradmu = (m.Pgal(k, mu+epsilon) - m.Pgal(k, mu-epsilon)) / (2.*epsilon)
         
        # derivative of AP remapping   
        apar2 = apar**2; aperp2 = aperp**2
        dk_da  = k * (mu**2 - 1) / aperp2 / (1. + mu**2 * (aperp2/apar2 - 1.))**0.5
        dmu_da = apar * mu * (mu**2 - 1.) / (-apar2 + (apar2-aperp2)*mu**2) /(1. + mu**2 * (aperp2/apar2 - 1.))**0.5
        
        # combine the 2nd term with chain rule
        term2 = 1./aperp**2/apar *  (gradk * dk_da + gradmu*dmu_da)
        
        return term1 + term2
