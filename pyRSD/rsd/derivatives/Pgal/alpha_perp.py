from . import PgalDerivative
from pyRSD.rsd import APLock

class dPgal_dalpha_perp(PgalDerivative):
    """
    The partial derivative of `Pgal` with respect to `alpha_perp`
    """
    param = 'alpha_perp'
    
    @staticmethod
    def eval(m, pars, k, mu, epsilon=1e-4):
        
        # some setup
        aperp = m.alpha_perp
        apar  = m.alpha_par
        F     = apar/aperp
        N     = (1. + mu**2 * (1/F**2 - 1.))  
                
        # derivative of the volume factor
        term1 = -2. * m.Pgal(k, mu) / aperp
        
        # first compute derivatives wrt to k, mu
        dPgal_dk = (m.Pgal(k+epsilon, mu) - m.Pgal(k-epsilon, mu)) / (2.*epsilon)
        dPgal_dmu = (m.Pgal(k, mu+epsilon) - m.Pgal(k, mu-epsilon)) / (2.*epsilon)
            
        # dPgal / dkprime
        dkprime_dk  = N**0.5 / aperp
        dPgal_dkprime  = dPgal_dk / dkprime_dk
    
        # compute dPgal / dmuprime
        dkprime_dmu    = (k*mu/aperp) * (1/F**2 -1.) / N**0.5
        dmuprime_dmu   = 1/F/N**1.5
        dPgal_dmuprime = (dPgal_dmu - dPgal_dkprime*dkprime_dmu) / dmuprime_dmu
        
        # derivative of remappings wrt alpha_perp  
        dkprime_da = -k*(1. - mu**2) / aperp**2 / N**0.5
        dmuprime_da = mu * (1. - mu**2) / (apar * N**1.5)

        return term1 + dPgal_dkprime * dkprime_da + dPgal_dmuprime * dmuprime_da
