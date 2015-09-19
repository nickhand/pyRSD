"""
 correlation.py
 pyRSD: module to compute the configuration space correlation function from a 
        power spectrum instance
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2014
"""
from ._cache import Cache, parameter, interpolated_property, cached_property
from .. import pygcl, numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.misc import derivative

KMAX = 100.


class ExtrapolatedMultipole(Cache):
    """
    Class to extrapolate the redshift space multipoles at low and high k, 
    using a power law at high k
    """
    def __init__(self, k, pole, ell, kcut=0.2, linear=False):
        
        """
        Parameters
        ----------
        power : pyRSD.rsd.power_dm
            An instance of the power spectrum to extrapolate
        
        kcut : float, optional    
            Beyond this wavenumber, use a power law extrapolation for the 
            power spectrum.
        
        ell : int, {0, 2}
            The multipole 
        
        linear : bool, optional
            Whether to do linear calculation
        """
        # initialize the Cache subclass first
        Cache.__init__(self)
        
        self.k = k
        self.pole = pole
        self.ell = ell                
        self.kcut = kcut

        # check kmin value
        if self.kmin_model > 0.05: 
            raise ValueError("Power spectrum must be computed down to at least k = 0.05 h/Mpc.")
            
        # check if kcut > kmax_model
        if (self.kcut > self.kmax_model and not self.linear):
            print "Warning: requested k_cut = %.2f is larger than maximum k for model, k = %.2f" %(self.kcut, self.kmax_model)
            self.kcut = self.kmax_model
    
    #---------------------------------------------------------------------------
    # attributes
    #---------------------------------------------------------------------------
    @parameter
    def k(self, val):
        """
        The k values where the input multipole is defined
        """
        return val
        
    @parameter
    def pole(self, val):
        """
        The input multipole values
        """
        return val
        
    @parameter
    def ell(self, val):
        """
        The multipole number
        """
        return val
    
    @parameter
    def b1(self, val):
        """
        The linear bias factor.
        """
        return val
        
    @parameter
    def f(self, val):
        """
        The value of the log growth rate f
        """
        return val
        
    @parameter
    def power_lin(self, val):
        """
        The linear power spectrum
        """
        return val
        
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("k")
    def kmin_model(self):
        """
        The minimum value of the input k
        """
        return np.amin(self.k)
    
    @cached_property("k")
    def kmax_model(self):
        """
        The maximum value of the input k
        """
        return np.amax(self.k)
        
    @cached_property("k", "pole")
    def model_spline(self):
        """
        Spline of the input (k, pole) values
        """
        return InterpolatedUnivariateSpline(self.k, self.pole)

            
    @cached_property("pole", "k", "kcut")
    def powerlaw_slope(self):
        """
        The power law extrapolation slop used beyond ``kcut``
        """
        inds = np.where(self.pole > 0.)
        logspline = InterpolatedUnivariateSpline(self.k[inds], np.log(self.pole[inds]))
        return derivative(logspline, self.kcut, dx=1e-3)*self.kcut
           
    def kaiser(self, k):
        """
        The linear, Kaiser power multipole, as specified by `self.ell`
        """
        beta = self.f/self.b1
        if (self.ell == 0):
            prefactor = 1. + (2./3.)*beta + (1./5.)*beta**2;
        elif (self.ell == 2):
            prefactor = (4./3.)*beta + (4./7.)*beta**2;

        return prefactor * self.b1**2 * self.power_lin(k)
    
    #---------------------------------------------------------------------------
    def _evaluate(self, k):
        """
        Internal function to evaluate
        """
        if (k < self.kmin_model):
            return self.pole[0]*self.kaiser(k)/self.kaiser(self.kmin_model)
        elif (k > self.kcut):
            return self.model_spline(self.kcut)*(k/self.kcut)**self.powerlaw_slope  
        else:
            return self.model_spline(k)
        
    def __call__(self, k):
        """
        Return the value at k
        """
        if np.isscalar(k):
            return self._evaluate(k)
        else:
            return np.array([self._evaluate(ki) for ki in k])
        

#-------------------------------------------------------------------------------

def XiMonopole(r, power, kcut=0.2, smoothing=0., linear=False):
    """
    Compute the smoothed correlation function monopole
    
    Parameters
    ----------
    r : array_like
        The separations to compute the correlation function monopole at [units: `Mpc/h`]
    power : pyRSD.rsd.power_biased
        The power spectrum class specifying the redshift-space power spectrum
        to integrate over
    kcut : float, optional
        The wavenumber (in `h/Mpc`) to integrate over. Default is 0.2 `h/Mpc`
    smoothing : float, optional
        The smoothing radius, default = 0.
    linear : bool, optional
        Use the linear, Kaiser redshift-space power multipoles
    """
    # make the extrapolated power object
    P_extrap = ExtrapolatedMultipole(power, kcut=kcut, ell=0, linear=linear)
    
    # set up the spline
    k_spline = np.logspace(-5, np.log10(KMAX), 1000)
    P_spline = P_extrap(k_spline)
    spline = pygcl.CubicSpline(k_spline, P_spline)
    
    
    return pygcl.SmoothedXiMultipole(spline, 0, r, 32768, 1e-5, KMAX, smoothing)
#end XiMonopole

#-------------------------------------------------------------------------------
def XiQuadrupole(r, power, kcut=0.2, smoothing=0., linear=False):
    """
    Compute the smoothed correlation function monopole
    
    Parameters
    ----------
    r : array_like
        The separations to compute the correlation function monopole at [units: `Mpc/h`]
    power : pyRSD.rsd.power_biased
        The power spectrum class specifying the redshift-space power spectrum
        to integrate over
    kcut : float, optional
        The wavenumber (in `h/Mpc`) to integrate over. Default is 0.2 `h/Mpc`
    smoothing : float, optional
        The smoothing radius, default = 0.
    linear : bool, optional
        Use the linear, Kaiser redshift-space power multipoles
    """
    # make the extrapolated power object
    P_extrap = ExtrapolatedMultipole(power, kcut=kcut, ell=0, linear=linear)
    
    # set up the spline
    k_spline = np.logspace(-5, np.log10(KMAX), 1000)
    P_spline = P_extrap(k_spline)
    spline = pygcl.CubicSpline(k_spline, P_spline)
    
    
    return -1*pygcl.SmoothedXiMultipole(spline, 2, r, 32768, 1e-5, KMAX, smoothing)
#end XiQuadrupole

#-------------------------------------------------------------------------------
    
    
    
