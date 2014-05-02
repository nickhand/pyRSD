"""
 correlation.pyx
 pyRSD: module to compute the configuration space correlation function from a 
        power spectrum instance
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2014
"""
from pyRSD.rsd cimport _fourier_integrals
import numpy as np

class Correlation(object):
    
    def __init__(self, power, kmin, kmax, smoothing_radius=0., kcut=0.2):
        
        """
        Parameters
        ----------
        power : pyRSD.rsd.power_dm
            An instance of the power spectrum to integrate over to compute
            the configuration space correlation function

        kmin : float
            The minimum wavenumber to integrate over [units: h/Mpc]
            
        kmax : float
            The maximum wavenumber to integrate over [units: h/Mpc]
        
        smoothing_radius : float, optional
            The radius over which to applying a Gaussian smoothing function
            [units: Mpc/h] Default is 0 (no smoothing).
        
        kcut : float, optional    
            Beyond this wavenumber, use a power law extrapolation for the 
            power spectrum.
        """
        self.power            = power
        self.kmin, self.kmax  = kmin, kmax
        self.smoothing_radius = smoothing_radius
        self.kcut             = kcut
    #end __init__
    
    #---------------------------------------------------------------------------    
    def _extrapolate_power(self, Pspec):
        """
        Internal function to do a power law extrapolation of the power spectrum
        at high wavenumbers.
        """
        if self.power.k.max() >= self.kmax:
            return self.power.k, Pspec
        else:
            # compute the logarithmic derivative, aka power law slope
            dlogP = np.diff(np.log(Pspec))
            dlogk = np.diff(np.log(self.power.k))
        
            imin = (np.abs(self.power.k-self.kcut)).argmin()
            slope = (dlogP / dlogk)[imin]
        
            inds = np.where(self.power.k < self.kcut)[0]
            k_extrap = np.linspace(self.kcut, self.kmax, 1000)
            k_full = np.concatenate( (self.power.k[inds], k_extrap) )
            P_full = np.concatenate( (Pspec[inds], Pspec[imin]*(k_extrap/self.kcut)**slope) )
        
            return k_full, P_full
    #end _extrapolate_power
    
    #---------------------------------------------------------------------------
    def monopole(self, s, linear=False):
        """
        Compute the monopole moment of the configuration space correlation 
        function.
        """
        # do the power law extrapolation past k = kcut
        self.k_extrap, self.P_extrap = self._extrapolate_power(self.power.monopole(linear=linear))
        
        # initialize the fourier integrals class
        integrals = _fourier_integrals.Fourier1D(0, self.kmin, self.kmax, 
                                                 self.smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)
                                                 
        return integrals.evaluate(s) 
    #end monopole
    
    #---------------------------------------------------------------------------
    def quadrupole(self, s):
        """
        Compute the monopole moment of the configuration space correlation 
        function.
        """
        # do the power law extrapolation past k = kcut
        k_extrap, P_extrap = self._extrapolate_power(self.power.quadrupole(linear=linear)))

        # initialize the fourier integrals class
        integrals = _fourier_integrals.Fourier1D(2, self.kmin, self.kmax, 
                                                 self.smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)

        return integrals.evaluate(s) 
    #end quadrupole
    
    #---------------------------------------------------------------------------
    
        
    
