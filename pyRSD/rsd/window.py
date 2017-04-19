import math
from fractions import Fraction
import numpy as np
import scipy.interpolate as interp

from pyRSD.rsd.tools import RSDSpline as spline
from pyRSD import pygcl
from pyRSD.rsd.grid_transfer import PkmuGrid, PolesTransfer

_epsilon = np.finfo(float).eps

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
    
    See Wilson et al 2015 for details
    """
    def __init__(self, s, W, max_ellprime=4, max_ell=4):
        """
        s : array_like, (Ns,)
            the separation vector
        W : array_like, (Ns, Nl)
            the even-ell configuration space window function multipoles, 
            where Nl must be >= 5; the first column is the ell=0, second 
            is ell=2, etc
        max_ellprime : int, optional
            the maximum value of ``ellprime`` to include when performing
            the linear combination of higher-order multipoles leaking
            into a mulitpole of order ``ell
        max_ell : int, optional
            maximum multipole number we want to convolve
        """
        # the values of the separation where window is defined
        self.s = s
        self.smin = s.min()
        self.smax = s.max()
        
        # the array of window multipoles
        self.W = W
        
        # ell and ell prime values
        self.max_ellprime = max_ellprime  
        self.max_ell      = max_ell
              
        # initialize the kernel splines
        self._setup_kernels()
        
    def _setup_kernels(self):
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
        splines = self.splines[ell]
        toret = np.zeros((len(r), len(splines)))
        
        idx = (r>=self.smin)&(r<=self.smax)
        for i, s in enumerate(splines):
            toret[idx,i] = s(r[idx])
            
            # set the kernel to one out of bounds
            if i == ell//2:
                toret[~idx,i] = 1.0
                
        return toret
        
    def __call__(self, ells, r, xi, order='F'):
        """
        Perform the linear combination of configuration-space multipoles
        with the kernel of window multipoles 
        
        Parameters
        ----------
        ells : list of int
            the list of multipole numbers that we are convolving
        r : array_like
            the desired separation vector where the configuration-space multipoles
            are defined
        xi : array_like, shape: (len(r), len(ells))
            the configuration-space multipoles
        order : 'F', 'C'
            memory-order of return array; 'C' is organized by rows, 'F' by columns
            
        Returns
        -------
        xi_conv : array_like
            the convolved xi arrays, given by a linear combination of ``xi`` and
            the window function multipoles
        """
        # convolved xi
        conv_xi = np.zeros((len(r), len(ells)), order=order)
        
        # convolve each ell
        for i, ell in enumerate(ells):
        
            # convolution kernel
            kern = self._get_kernel(ell, r)
            
            # check shapes
            if kern.shape[1] != xi.shape[1]:
                npoles = self.max_ellprime//2+1
                
                # need at least a shape of npoles
                if xi.shape[1] > npoles:
                    xi = xi[...,:npoles]
                else:
                    raise ValueError(("shape mismatch between kernel and number of xi multipoles; "
                                      "please provide the first %d even multipoles" %npoles))
            
            conv_xi[:,i] = np.einsum('ij,ij->i', xi, kern)
            
        return conv_xi  
     
class WindowTransfer(PolesTransfer):
    """
    Class to handle the transfer function associated with the window 
    function convolution of the power spectrum multipoles
    
    Notes
    -----
    *   this class DOES NOT take in to the effects of discrete binning
    *   it uses a log-spaced hi-resolution grid to compute the FFTs needed 
        in the convolution
    *   it either returns at the ``k`` values of the hi-res grid, or optionally,
        the values of ``k_out`` supplied to the constructor
    """
    def __init__(self, window, ells, 
                    grid_kmin=1e-4, 
                    grid_kmax=100., 
                    Nk=1024, 
                    Nmu=40, 
                    kmax=0.7, 
                    max_ellprime=4, 
                    power=None, 
                    ells_mask=None,
                    k_out=None):
        """
        Parameters
        ----------
        window : array_like
            the window function multipoles in configuration space, as columns;
            the first column should be the separation vector ``s``
        grid_kmin : float, optional
            the minimum ``k`` value on the grid used when FFTing during the convolution
        grid_kmax : float, optional
            the maximum ``k`` value on the grid used when FFTing during the convolution
        Nk : int, optional
            the number of grid points in the ``k`` direction
        Nmu : int, optional
            the number of grid points in the ``mu`` direction (from 0 to 1)
        kmax : int, optional
            the maximum wavenumber to evaluate the model at
        max_ellprime : int, optional
            the maximum multipole number to include when determining the leakage
            of higher-order multipoles into a multipole of order ``ell``
        power : array_like
            the power values defined on the grid -- can have shape
            (Nk,Nmu) or (N,), in which case it is interpreted
            as the values at all valid grid points
        ells_mask : list, None, optional
            boolean mask specifying which ell values to return; this is needed in the 
            case that we are doing a window convolution and need more ell values
        k_out : array_like; optional
            return the multipoles at these k values
        """        
        # make the grid
        k = np.logspace(np.log10(grid_kmin), np.log10(grid_kmax), Nk)
        mu_edges = np.linspace(0., 1., Nmu+1)
        mu = 0.5 * (mu_edges[1:] + mu_edges[:-1])
        grid_k, grid_mu =  np.meshgrid(k, mu, indexing='ij')
        
        weights = np.ones_like(grid_k)
        weights[grid_k > kmax] = 0.
        grid = PkmuGrid([k,mu], grid_k, grid_mu, weights)
        
        # init the base class
        super(WindowTransfer, self).__init__(grid, ells, kmin=grid_kmin, kmax=kmax, power=power, ells_mask=ells_mask)
                
        # the convolver object
        self.convolver = WindowConvolution(window[:,0], window[:,1:], max_ellprime=max_ellprime, max_ell=int(max(ells)))
    
        # the output k
        self.k_out = k_out
    
    def __call__(self, flatten=False, qbias=0.7, dry_run=False, no_convolution=False):
        """
        Do the convolution
        
        Parameters
        ----------
        flatten : bool; optional
            return the array flattened, column-wise
        qbias : float; optional
            when using FFTLog to perform the convolution, bias the transform
            with a power-law of this value
        dry_run : bool; optional
            if ``True``, do not perform the convolution linear combinination
            in configuration space; mostly for debugging purposes in order
            to check the reversability of the transform
        no_convolution : bool; optional
            do not perform the convolution at all (no FFTs)
        
        Returns
        -------
        Pell_conv : array_like
            the convolved multipoles
        """
        # zero-pad beyond the accuracy of the model
        self.power[~self.grid.notnull] = 0.
        
        # do the legendre-weighted sum of self.power
        Pell = super(WindowTransfer, self).__call__(flatten=False, full_grid_output=True)
        Pell = np.nan_to_num(Pell)
        
        # do the convolution
        if not no_convolution:

            # FFT the input power multipoles
            xi = np.empty((self.grid.Nk, len(self.ells)), order='F') # column-continuous
            rr = np.empty(self.grid.Nk)
            for i, ell in enumerate(self.ells):
                pygcl.ComputeXiLM_fftlog(int(ell), 2, self.grid.k_cen, Pell[:,i], rr, xi[:,i], qbias)
                xi[:,i] *= (-1)**(ell//2)

            # the linear combination of multipoles
            if dry_run:
                xi_conv = xi.copy()
            else:
                xi_conv = self.convolver(self.ells, rr, xi, order='F')

            # FFTLog back
            Pell_conv = np.empty((self.grid.Nk, len(self.ells)), order='F')
            kk = np.empty(self.grid.Nk)
            for i, ell in enumerate(self.ells):
                pygcl.ComputeXiLM_fftlog(int(ell), 2, rr, xi_conv[:,i], kk, Pell_conv[:,i], -qbias)
                Pell_conv[:,i] *= (-1)**(ell//2) * (2*np.pi)**3
        else:
            Pell_conv = Pell
          
        # restrict to the correct k values  
        Pell_conv = self.restrict_k(Pell_conv)
    
        # interpolate to k_out
        if self.k_out is not None:
            if self.k_out.shape[1] != Pell_conv.shape[1]:
                raise ValueError("shape mismatch in second dimension of ``k_out``")
                    
            toret = np.ones_like(self.k_out) * np.nan
            for i, ell in enumerate(self.ells[self.ells_mask]):
                
                kk = self.coords[0][:,i]
                idx = np.isfinite(kk)
                spl = spline(kk[idx], Pell_conv[idx,i])
                
                idx = np.isfinite(self.k_out[:,i])
                toret[idx,i] = spl(self.k_out[idx,i])
        else:
            toret = Pell_conv 

        # flatten
        if flatten:
            toret = toret.ravel(order='F')
            toret = toret[np.isfinite(toret)]

        return toret  
        
def convolve_multipoles(k, Pell, ells, convolver, qbias=0.7, dry_run=False, legacy=True):
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
    if not legacy:
        
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
        
    else:
        
        shape = Pell.shape
        Nell = len(ells)
        if Nell != shape[-1]:
            raise ValueError("shape mismatch between multipole numbers and number of multipoles provided")
        
        if not all(ell in [0,2,4,6] for ell in ells):
            raise ValueError("valid `ell` values are [0,2,4,6]")
    
        # separation is the first window column
        s = convolver.s
    
        # format the k_out
        k_out = k
        if np.ndim(k_out) == 1:
            k_out = np.repeat(k_out[:,None], Nell, axis=1)
        if k_out.shape[-1] != len(ells):
            raise ValueError("input `k_out` must have %d columns for ell=%s multipoles" %(Nell, str(ells)))
    
        # make the hires version to avoid wiggles when convolving
        if len(k) < 500:
            k_hires = np.logspace(np.log10(k.min()), np.log10(k.max()), 500)
            poles_hires = []
            for i in range(Nell):
                tck = interp.splrep(k, Pell[:,i], k=3, s=0)
                poles_hires.append(interp.splev(k_hires, tck))
            Pell = np.vstack(poles_hires).T
            k = k_hires.copy()
        
        # FT the power multipoles    
        xi = np.empty((len(s), Nell))
        for i, ell in enumerate(ells): 
            xi[:,i] = pygcl.pk_to_xi(int(ell), k, Pell[:,i], s, smoothing=0., method=pygcl.IntegrationMethods.TRAPZ)
    
        # convolve the config space multipole
        if dry_run:
            xi_conv = xi.copy()
        else:
            xi_conv = convolver(ells, s, xi, order='F')
    
        # FT back to get convolved power pole
        toret = np.empty((len(k_out), Nell))
        for i, ell in enumerate(ells):
            toret[:,i] = pygcl.xi_to_pk(int(ell), s, xi_conv[:,i], k_out[:,i], smoothing=0., method=pygcl.IntegrationMethods.TRAPZ)
    
        return k_out[:,0], toret 