"""
 integralsIJ.py
 pyPT: integral I_nm(k), J_nm(k) from Appendix D of Vlah et al. 2012
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
import kernels
import numpy as np
import scipy as sp
import utils.samplinator as s
from utils import pytools
import pylab 

class Inm(s.with_sampleable_methods):
    """
    Integral I_nm(k) from Appendix D of Vlah et al. 2012.
    """
    
    def __init__(self, n, m, Plin_func, kmin=1e-3, kmax=1.):
        """
        Parameters
        ----------
        n : int
            Determines the first index of the kernel. Must be < 4.
        m : int
            Determines the second index of the kernel. Must be < 4.
        Plin_func : callable
            Python function that gives the linear matter power spectrum. It must
            take a single variable k, defined to be the wavenumber
        kmin : float, optional
            the minimum wavenumber to integrate over; default is 1e-3.
        kmax : float, optional
            the maximum wavenumber to integrate to; default is 10. 
        """
        if n > 3 or m > 3: 
            raise ValueError("kernel f_nm must have n, m < 4")
        self.kernel = kernels.f[n, m]
        
        self.kmin = kmin
        self.kmax = kmax
        self.Plin = Plin_func
    #end __init__
    
    #---------------------------------------------------------------------------
    def _integrand(self, lnq, x, k):
        """
        Internal method defining the integrand to use. 
        """
        q = np.exp(lnq)
        k_minus_q = np.sqrt(q*q + k*k - 2.*k*q*x)
        fac = 1./(2*np.pi)**2
        
        p = fac*q**3*self.kernel(q/k, x)*self.Plin(q)*self.Plin(k_minus_q)
        return p
    #end _integrand
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def __call__(self, k):
        """
        Make the class callable and return the double integration. 
        """

        ans, err = sp.integrate.dblquad(self._integrand, -1., 1., 
                                        lambda x: np.log(self.kmin),
                                        lambda x: np.log(self.kmax),
                                        args=(k,), 
                                        epsrel = 1e-4, epsabs = 1e-8)
        # x = np.linspace(-1, 0.9999, 50)
        # q = np.logspace(-3, 0, 1000)
        # lnq = np.log(q)
        # for ix in x:
        #     p = self._integrand(lnq, ix, k)
        #     pylab.plot(lnq, p)
        #     pylab.title("x = %f" %ix)
        #     pylab.show()
        
        return ans
    #end __call__
    
    #---------------------------------------------------------------------------
#endclass Inm
   
#-------------------------------------------------------------------------------     
class Jnm(s.with_sampleable_methods):
    """
    Integral J_nm(k) from Appendix D of Vlah et al. 2012.
    """
    
    def __init__(self, n, m, Plin_func, kmin=0.01, kmax=10.):
        """
        Parameters
        ----------
        n : int
            Determines the first index of the kernel. Must be < 4.
        m : int
            Determines the second index of the kernel. Must be < 4.
        Plin_func : callable
            Python function that gives the linear matter power spectrum. It must
            take a single variable k, defined to be the wavenumber
        kmin : float, optional
            the minimum wavenumber to integrate over; default is 1e-3.
        kmax : float, optional
            the maximum wavenumber to integrate to; default is 10. 
        """
        if (n + m > 2): 
            raise ValueError("kernel g_nm must have n, m such that n + m < 3")
        self.kernel = kernels.g[n, m]
        
        self.kmin = kmin
        self.kmax = kmax
        self.Plin = Plin_func
    #end __init__
    
    #---------------------------------------------------------------------------
    def _integrand(self, lnq, k):
        """
        Internal method defining the integrand to use. 
        """
        q = np.exp(lnq)
        fac = 1./(2*np.pi**2)
        return fac*q*self.kernel(q/k)*self.Plin(q)
        
    #end _integrand
    
    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Make the class callable and return the double integration. 
        """
        ans, err = sp.integrate.quad(self._integrand, np.log(self.kmin), 
                                    np.log(self.kmax), args=(k,), 
                                    epsrel = 1e-5, epsabs = 1e-5)
        return ans
    #end __call__
    
    #----------------------------------------------------------------------------
#endclass Jnm
