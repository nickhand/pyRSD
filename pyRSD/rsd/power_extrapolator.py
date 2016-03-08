from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.optimize as opt

from .. import numpy as np
from . import tools
from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator

#------------------------------------------------------------------------------
# extrapolation function
#------------------------------------------------------------------------------
def power_law_extrapolation(x, alpha, beta=0.):
    """
    Power law extrapolation function, with a slope that can
    optionally vary with independent variable
    
    :math: f(x, alpha, beta) = x^{alpha + beta*log10(x)}
    
    Parameters
    ----------
    x : float, array_like
        the (normalized) array of independent variables
    alpha : float
        the constant power-law slope
    beta : float, optional (`0`)
        the variable power-law slope, default is `0`
    """
    return x**(alpha + beta*np.log10(x))
    

#------------------------------------------------------------------------------
class ExtrapolatedPowerSpectrum(Cache):
    """
    Class to extrapolate a power spectrum model as a function 
    of (k, mu), using a constant power-law extrapolation at low k and a 
    variable-slope power law at high k
    """
    def __init__(self, model, model_func='Pgal', **kwargs):
        """
        Parameters
        ----------
        model : rsd.DarkMatterSpectrum or its derived classes
            the power spectrum model instance
        model_func : str, optional (`Pgal`)
            the name of the function, which is a class method of ``model``
            and takes (k, mu) as its arguments
        kwargs : various keyword parameters
            k_lo : float 
                wavenumber to extrapolate below; default is ``model.kmin``
            k_hi : float
                wavenumber to extrapolate above; default is ``model.kmax``
            kcut_lo : float
                only include wavenumbers below this value when fitting
                the extrapolation parameters; default is `5*k_lo`
            kcut_hi : float
                only include wavenumbers above this value when fitting
                the extrapolation parameters; default is `0.5*k_hi`
        """        
        # the model
        self.model = model
        self.model_kmin = model.kmin
        self.model_kmax = model.kmax
        self.model_func = model_func
        
        # k values for lo/hi extrapolation
        self.k_hi = kwargs.get('k_hi', self.model.kmax)
        self.k_lo = kwargs.get('k_lo', self.model.kmin)
        
        # extrapolation ranges
        self.kcut_hi = kwargs.get('kcut_hi', 0.5*self.k_hi)
        self.kcut_lo = kwargs.get('kcut_lo', 5.*self.k_lo)
         
    #------------------------------------------------------------------------------
    # parameters
    #------------------------------------------------------------------------------
    @parameter
    def model_func(self, val):
        """
        The name of the function, which is a class method of ``model``
        and takes (k, mu) as its arguments
        """
        if not hasattr(self.model, val):
            raise ValueError("``model`` has not class method ``%s``" %val)
        return val
        
    @parameter
    def k_hi(self, val):
        """
        The wavenumber value, above which we will extrapolate the power
        spectrum with a power law of varying slope
        """
        if val > self.model_kmax*0.95:
            raise ValueError("`k_hi` cannot be greater than `0.95*model_kmax`")
        return val
        
    @parameter
    def k_lo(self, val):
        """
        The wavenumber value, above which we will extrapolate the power
        spectrum with a power law of varying slope
        """
        if val < self.model_kmin*1.05:
            raise ValueError("`k_lo` cannot be less than `1.05*model_kmin`")
        return val
        
    @parameter
    def kcut_hi(self, val):
        """
        Only include wavenumbers above this value when fitting the high-k
        extrapolation parameters
        """
        if val >= self.k_hi:
            raise ValueError("`kcut_hi` must be less than `k_hi`")
        return val
        
    @parameter
    def kcut_lo(self, val):
        """
        Only include wavenumbers below this value when fitting the low-k
        extrapolation parameters
        """
        if val <= self.k_lo:
            raise ValueError("`kcut_lo` must be greater than `k_lo`")
        return val
        
    @parameter
    def model_kmin(self, val):
        """
        The minumum wavenumber that the model can compute
        """
        self.model.kmin = val
        return val
        
    @parameter
    def model_kmax(self, val):
        """
        The maximum wavenumber that the model can compute
        """
        self.model.kmax = val
        return val
        
    #------------------------------------------------------------------------------
    # cached properties
    #------------------------------------------------------------------------------
    @cached_property("model_kmin", "model_kmax")
    def _ks(self):
        """
        Internal variable that defines wavenumbers on a grid for interpolating
        """
        return np.logspace(np.log10(self.model_kmin*1.05), np.log10(self.model_kmax*0.95), 200)
        
    @cached_property()
    def _mus(self):
        """
        Internal variable that defines mus on a grid for interpolating
        """
        return np.linspace(0, 1., 100)
        
    @cached_property("_ks", "_mus")
    def _grid(self):
        """
        The (k,mu) grid for interpolating -- stacked such that
        the output shape is (Nk, Nmu, 2)
        """
        ks, mus = np.broadcast_arrays(self._ks[:,None], self._mus[None,:])
        return np.dstack([ks, mus])
      
    @cached_property("_ks", "_mus")
    def _power_spline(self):
        """
        The interpolating for ``model.Pgal``, interpolated using the (k,mu) grid
        """  
        power = self.model.Pgal(self._ks, self._mus)
        f = RegularGridInterpolator([self._ks, self._mus], power)
        return f
        
    @cached_property("k_hi", "kcut_hi", "_grid")
    def _high_k_splines(self):
        """
        The splines for ``(alpha, beta)`` vs `_mus` for the high-k extrapolation
        """
        mask = (self._ks >= self.kcut_hi)
        params = []
        for i, mu in enumerate(self._mus):
            
            # define the objective function for this mu
            amplitude = self._power_spline([self.k_hi, mu])[0]
            objective = lambda x, alpha, beta: amplitude * power_law_extrapolation(x, alpha, beta)
            
            # normalized x and y values
            x = self._ks[mask]/self.k_hi
            y = self._power_spline(self._grid[mask, i, :])
            
            # do the fit
            popt, pcov = opt.curve_fit(objective, x, y)
            params.append(popt)
            
        params = np.asarray(params)
        toret = []
        for p in params.T:
            toret.append(spline(self._mus, p))
        return toret
    
    @cached_property("k_lo", "kcut_lo", "_grid")
    def _low_k_splines(self):

        mask = (self._ks <= self.kcut_lo)
        params = []
        for i, mu in enumerate(self._mus):

            # define the objective function for this mu
            amplitude = self._power_spline([self.k_lo, mu])[0] 
            objective = lambda x, alpha: amplitude * power_law_extrapolation(x, alpha, beta=0)
            
            # normalized x and y values
            x = self._ks[mask]/self.k_lo
            y = self._power_spline(self._grid[mask, i, :])
            
            # do the fit
            popt, pcov = opt.curve_fit(objective, x, y)
            params.append(popt)

        params = np.asarray(params)
        return spline(self._mus, params[:,0])
                    
    #------------------------------------------------------------------------------
    # the main functions
    #------------------------------------------------------------------------------
    def high_k_params(self, mu):
        """
        Return a tuple (`alpha`, `beta`) for this `mu` value for use in the
        high-k extrapolation
        """
        return tuple(f(mu) for f in self._high_k_splines)
        
    def low_k_params(self, mu):
        """
        Return `alpha` for this `mu` value for use in the low-k extrapolation
        (`beta` = 0 by assumption)
        """
        return self._low_k_splines(mu)
        
    @tools.broadcast_kmu
    def __call__(self, k, mu):
        """
        Return the power as a function of (k, mu), using the power-law 
        extrapolations when `k` is less than `k_lo` or greater than
        `k_hi`
        
        Note
        ----
        *   this broadcasts the result if `k` and `mu` are both 
            array_like with different shapes, such that the result
            will have shape (`Nk`, `Nmu`)
        
        Parameters
        ----------
        k : float, array_like
            the wavenumbers to compute the power at
        mu : float, array_like
            the mu values to compute the power at
        """
        k, mu = np.broadcast_arrays(k, mu)
        toret = np.empty(k.shape)
        
        # the relevant indices
        lo_idx = (k < self.k_lo)
        mid_idx = (k >= self.k_lo)&(k <= self.k_hi)
        hi_idx = (k > self.k_hi)
        
        # low k
        if lo_idx.sum():
            alpha = self.low_k_params(mu[lo_idx])
            x = k[lo_idx]/self.k_lo
            toret[lo_idx] = self.model.Pgal(self.k_lo, mu[lo_idx]) * power_law_extrapolation(x, alpha)
         
        # mid k  
        if mid_idx.sum():
            toret[mid_idx] = np.squeeze(self.model.Pgal(k[mid_idx], mu[mid_idx]))
        
        # high k
        if hi_idx.sum():
            alpha, beta = self.high_k_params(mu[hi_idx])
            x = k[hi_idx]/self.k_hi
            toret[hi_idx] = self.model.Pgal(self.k_hi, mu[hi_idx]) * power_law_extrapolation(x, alpha, beta)

        return toret
        
    def to_poles(self, k, ells, Nmu=41, flatten=False):
        """
        Compute the multipoles by integrating over the extrapolated
        power spectrum
        """
        from scipy.special import legendre
        from scipy.integrate import simps
        
        scalar = np.isscalar(ells)
        if scalar: ells = [ells]
    
        mus = np.linspace(0., 1., Nmu)
        pkmu = self(k, mus)
    
        if len(ells) != len(k):
            toret = []
            for ell in ells:
                kern = (2*ell+1.)*legendre(ell)(mus)
                val = np.array([simps(kern*d, x=mus) for d in pkmu])
                toret.append(val)
                      
            if scalar:
                return toret[0]
            else:
                toret = np.vstack(toret).T
                return toret if not flatten else np.ravel(toret, order='F')
        else:
            kern = np.asarray([(2*ell+1.)*legendre(ell)(mus) for ell in ells])
            return np.array([simps(d, x=mus) for d in kern*pkmu])