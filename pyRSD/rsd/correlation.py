"""
 correlation.py
 pyRSD: module to compute the configuration space correlation function from a 
        power spectrum instance
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2014
"""
from .. import pygcl, numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.misc import derivative

KMAX = 100.


class ExtrapolatedMultipole(object):
    """
    Class to extrapolate the redshift space multipoles at low and high k, 
    using a power law at high k
    """
    
    def __init__(self, power, kcut=0.2, ell=0, linear=False):
        
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
        self.power = power
        self.kcut = kcut
        self.linear = linear
                
        assert (ell in [0, 2]), "Multipole number must be 0 or 2"
        self.ell = ell

        if self.linear:
            self.Pspec = self.kaiser(self.power.k)
        else:
            if (self.ell == 0):
                self.Pspec = self.power.monopole()
            else:
                self.Pspec = self.power.quadrupole()
        
        # check kmin value
        if self.kmin_model > 0.05: 
            raise ValueError("Power spectrum must be computed down to at least k = 0.05 h/Mpc.")
            
        # check if kcut > kmax_model
        if (self.kcut > self.kmax_model and not self.linear):
            print "Warning: requested k_cut = %.2f is larger than maximum k for model, k = %.2f" %(self.kcut, self.kmax_model)
            self.kcut = self.kmax_model
        
    #end __init__
    
    #---------------------------------------------------------------------------
    # Some attributes we need
    #---------------------------------------------------------------------------
    @property
    def b1(self):
        """
        The linear bias factor.
        """
        return self.power.b1
            
    @b1.setter
    def b1(self, val):
        self.power.b1 = val
                        
        # delete the linear kaiser splines
        for a in ['_kaiser_mono_spline', '_kaiser_quad_spline']:
            if hasattr(self, a): delattr(self, a)
    
    #---------------------------------------------------------------------------
    @property
    def kmin_model(self):
        return np.amin(self.power.k)
    
    #---------------------------------------------------------------------------
    @property
    def kmax_model(self):
        return np.amax(self.power.k)
        
    #---------------------------------------------------------------------------
    @property
    def model_spline(self):
        try:
            return self._model_spline
        except AttributeError:
            if (self.ell == 0):
                self._model_spline = InterpolatedUnivariateSpline(self.power.k, self.Pspec)
            elif (self.ell == 2):
                self._model_spline = InterpolatedUnivariateSpline(self.power.k, self.Pspec)
            return self._model_spline
            
    #---------------------------------------------------------------------------
    @property
    def powerlaw_slope(self):
        try:
            return self._powerlaw_slope
        except AttributeError:
            inds = np.where(self.Pspec > 0.)
            logspline = InterpolatedUnivariateSpline(self.power.k[inds], np.log(self.Pspec[inds]))
            self._powerlaw_slope = derivative(logspline, self.kcut, dx=1e-3)*self.kcut
            return self._powerlaw_slope
            
    #---------------------------------------------------------------------------
    def kaiser(self, k):
        """
        The linear, Kaiser power multipole, as specified by `self.ell`
        """
        beta = self.power.f/self.b1
        if (self.ell == 0):
            prefactor = 1. + (2./3.)*beta + (1./5.)*beta**2;
        elif (self.ell == 2):
            prefactor = (4./3.)*beta + (4./7.)*beta**2;
            
        return prefactor * self.b1**2 * self.power.normed_power_lin(k)
    #end kaiser
    
    #---------------------------------------------------------------------------
    def _evaluate(self, k):
        
        if self.linear:
            return self.kaiser(k)
        
        if (k < self.kmin_model):
            return self.Pspec[0]*self.kaiser(k)/self.kaiser(self.kmin_model)
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
#endclass ExtrapolatedMultipole

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
    
    
    
