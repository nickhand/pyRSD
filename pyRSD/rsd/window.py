import math
from fractions import Fraction
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from pyRSD import pygcl

def G(p):
    """
    Return the function G(p), as defined in Wilson et al 2015.
    
    See also: WA Al-Salam 1953
    
    Returns
    -------
    numer, denom: int
        the numerator and denominator
    """
    toret = 1
    for p in range(p+1):
        if p == 0:
            toret *= 1.
        else:
            toret *= (1/2 + p - 1)
    return int(2**p) * toret, math.factorial(p)
    
def get_coefficients(ell, ellprime, as_string=False):
    """
    Return the window convolution coefficients
    
    Parameters
    ----------
    ell : int
        the multipole number of the spectra we are convolving
    ellprime : int
        the multipole number of the spectra that is leaking
        power via the convolution
    """
    p = 0
    coeffs = []
    qvals = []
    ret_str = []
    for p in range(0, min(ell, ellprime)+1):
        
        numer = []
        denom = []
        
        # numerator of product of G(x)
        for r in [G(ell-p), G(p), G(ellprime-p)]:
            numer.append(r[0])
            denom.append(r[1])
            
        # divide by this
        a,b = G(ell+ellprime-p)
        numer.append(b)
        denom.append(a)
        
        numer.append((2*(ell+ellprime) - 4*p + 1))
        denom.append((2*(ell+ellprime) - 2*p + 1))
    
        q = ell+ellprime-2*p
        numer.append((2*ell+1))
        denom.append((2*q+1))
        
        numer = Fraction(np.prod(numer))
        denom = Fraction(np.prod(denom))
        if not as_string:
            coeffs.append(float(numer/denom))
            qvals.append(q)
        else:
            ret_str.append("%s L%d" %(numer/denom, q))
        
    if not as_string:
        return qvals[::-1], coeffs[::-1]
    else:
        return ret_str[::-1]
        
class WindowConvolution(object):
    """
    Object to compute the window-convolved configuration space multipoles, 
    from the ell = 0, 2, 4 (,6) unconvolved multipoles and the window
    """
    def __init__(self, s, W, max_ellprime=4, max_ell=4):
        """
        s : array_like, (Ns,)
            the separation vector
        W : array_like, (Ns, Nl)
            the even-ell configuration space window function multipoles, 
            where Nl must be >= 5; the first column is the ell=0, second 
            is ell=2, etc
        """
        self.s = s
        self.W = W
        self.max_ellprime = max_ellprime  
        self.max_ell = max_ell
              
        self._initialize_splines()
        
    def _initialize_splines(self):
        """
        Initialize the splines used to compute the convolution
        kernels for each ell from the discretely-measued
        window multipoles
        """
        self.splines = {}
        kern = np.zeros((len(self.s), self.max_ellprime//2+1))
        W = self.W
        
        # ell is the multipole number of the convolved spectra
        for i, ell in enumerate(range(0, self.max_ell+1, 2)):
            
            # ellprime specifies power leakage from other multipoles into ell
            for j, ellprime in enumerate(range(0, self.max_ellprime+1, 2)):
                
                # the coefficients
                qvals, coeffs = get_coefficients(ell, ellprime)
                qinds = [q//2 for q in qvals]
                
                # this term is the sum of coefficients times the window multipoles
                kern[:,j] = np.einsum('...i,i...', W[:,qinds], np.array(coeffs))
                
            # store a spline representation
            self.splines[ell] = [spline(self.s, k) for k in kern.T]
    
    def _get_kernel(self, ell, r):
        """
        Return the appropriate kernel
        """
        return np.vstack([s(r) for s in self.splines[ell]]).T
        
    def __call__(self, ells, r, xi, order='F'):
        """
        Convolve the input configuration space multipoles with the window
        """
        # convolved xi
        conv_xi = np.zeros((len(r), len(ells)), order=order)
        
        # convolve each ell
        for i, ell in enumerate(ells):
        
            # convolution kernel
            kern = self._get_kernel(ell, r)
            if kern.shape[1] != xi.shape[1]:
                npoles = self.max_ellprime//2+1
                raise ValueError(("shape mismatch between kernel and number of xi multipoles; "
                                  "please provide the first %d even multipoles" %npoles))
            
            conv_xi[:,i] = np.einsum('ij,ij->i', xi, kern)
        
        return conv_xi   
    
def convolve_multipoles(k, Pell, ells, convolver, qbias=0.7, dry_run=False):
    """
    Convolve the input ell = 0, 2, 4 power multipoles, specified by `Pell`,
    with the specified window function.
    
    Parameters
    ----------
    k : array_like, (Nk,)
        the array of wavenumbers where `Pell` is defined -- to avoid convolution
        errors, `k` should probably extend to higher values than the desired `k_out`
    ells : array_like, (Nell,)
        the ell values
    Pell : array_like, (Nk, Nell)
        the ell = 0, 2, 4 power multipoles, defined at `k`
    """
    Nell = len(ells); Nk = len(k)
    
    # FFT the input power multipoles
    xi = np.empty((Nk, Nell), order='F') # column-continuous
    rr = np.empty(Nk)
    
    for i, ell in enumerate(ells):
        pygcl.ComputeXiLM_fftlog(int(ell), 2, k, Pell[:,i], rr, xi[:,i], qbias)
        xi[:,i] *= (-1)**(ell//2)
    
    # convolve
    if dry_run:
        xi_conv = xi.copy()
    else:
        xi_conv = convolver(ells, rr, xi, order='F')

    # FFTLog back
    Pell_conv = np.empty((Nk, Nell), order='F')
    kk = np.empty(Nk)
    for i, ell in enumerate(ells):
        pygcl.ComputeXiLM_fftlog(int(ell), 2, rr, xi_conv[:,i], kk, Pell_conv[:,i], -qbias)
        Pell_conv[:,i] *= (-1)**(ell//2) * (2*np.pi)**3
    
    return kk, Pell_conv   
