"""
 correlation.pyx
 pyRSD: module to compute the configuration space correlation function from a 
        power spectrum instance
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2014
"""
from pyRSD.rsd cimport _fourier_integrals
from pyRSD.cosmology import _functionator
import numpy as np

KMAX = 100.

class Correlation(object):
    
    def __init__(self, power, smoothing_radius=0., kcut=0.2):
        
        """
        Parameters
        ----------
        power : pyRSD.rsd.power_dm
            An instance of the power spectrum to integrate over to compute
            the configuration space correlation function
        
        smoothing_radius : float, optional
            The radius over which to applying a Gaussian smoothing function
            [units: Mpc/h] Default is 0 (no smoothing).
        
        kcut : float, optional    
            Beyond this wavenumber, use a power law extrapolation for the 
            power spectrum.
        """
        self.power            = power
        self.smoothing_radius = smoothing_radius
        self.kcut             = kcut
    #end __init__
    
    #---------------------------------------------------------------------------    
    def _extrapolate_power(self, Pspec, kmin):
        """
        Internal function to do a power law extrapolation of the power spectrum
        at high wavenumbers.
        """
        k = self.power.k
        
        # do a linear theory extrapolation at low k, if needed
        if k.min() > kmin: 
            
            # raise an exception if we haven't computed the spectrum down to 
            # at least k = 0.05
            if k.min() > 0.05: 
                raise ValueError("Power spectrum must be computed down to at least k = 0.05 h/Mpc.")
            
            # check that the linear power spectrum is evaluated at the k we need
            if Pspec.integrals.klin.min() > kmin:
                raise ValueError("Minimum wavenumber needed for convergence too low.")
                
            lowk_interp = _functionator.splineInterpolator(self.power.integrals.klin, self.power.integrals.Plin)
            lowk        = np.logspace(np.log10(kmin), np.log10(k.min()), 200)[:-1]
            lowP        = lowk_interp(lowk)
            
            k     = np.concatenate( (lowk, k) )
            Pspec = np.concatenate( (lowP, Pspec) )
        
        # also, do a power law extrapolation at high k, if needed
        if k.max() < KMAX:
            
            # compute the logarithmic derivative, aka power law slope
            dlogP = np.diff(np.log(Pspec))
            dlogk = np.diff(np.log(k))
        
            imin  = (np.abs(k - self.kcut)).argmin()
            slope = (dlogP / dlogk)[imin]
        
            inds     = np.where(k < self.kcut)[0]
            k_extrap = np.linspace(self.kcut, KMAX, 200)
            k        = np.concatenate( (k[inds], k_extrap) )
            Pspec    = np.concatenate( (Pspec[inds], Pspec[imin]*(k_extrap/self.kcut)**slope) )
        
        return k, Pspec
    #end _extrapolate_power
    
    #---------------------------------------------------------------------------
    def monopole(self, s, linear=False):
        """
        Compute the monopole moment of the configuration space correlation 
        function.
        """
        # compute the minimum wavenumber we need
        kmin = 0.1 / np.amax(s)
        
        # do the power law extrapolation past k = kcut
        self.k_extrap, self.P_extrap = self._extrapolate_power(self.power.monopole(linear=linear), kmin)
        
        # initialize the fourier integrals class
        integrals = _fourier_integrals.Fourier1D(0, self.kmin, self.smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)
                                                 
        return integrals.evaluate(s) 
    #end monopole
    
    #---------------------------------------------------------------------------
    def quadrupole(self, s, linear=False):
        """
        Compute the monopole moment of the configuration space correlation 
        function.
        """
        # do the power law extrapolation past k = kcut
        k_extrap, P_extrap = self._extrapolate_power(self.power.quadrupole(linear=linear))

        # initialize the fourier integrals class
        integrals = _fourier_integrals.Fourier1D(2, self.kmin, self.kmax, 
                                                 self.smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)

        return -1.*integrals.evaluate(s) 
    #end quadrupole
    
    #---------------------------------------------------------------------------
    
        
    
