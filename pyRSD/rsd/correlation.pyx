"""
 correlation.pyx
 pyRSD: module to compute the configuration space correlation function from a 
        power spectrum instance
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2014
"""
from pyRSD.rsd cimport _fourier_integrals
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from scipy.misc import derivative

KMAX = 100.

class Correlation(object):
    
    def __init__(self, power):
        
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
        self.power = power
    #end __init__
    #---------------------------------------------------------------------------
    def _kaiser_monopole(self, k):
        
        try:
            return self._kaiser_mono_spline(k)
        except:
            beta = self.power.f/self.power.b1
            mono_linear = (1. + 2./3*beta + 1/5*beta**2) * self.power.b1**2 * self.power.integrals.Plin
            self._kaiser_mono_spline = InterpolatedUnivariateSpline(self.power.integrals.klin, mono_linear)
            return  self._kaiser_mono_spline(k)
    #end _kaiser_monopole
    
    #---------------------------------------------------------------------------
    def _kaiser_quadrupole(self, k):
        
        try:
            return self._kaiser_quad_spline(k)
        except:
            beta = self.power.f/self.power.b1
            quad_linear = (4./3*beta + 4./7*beta**2) * self.power.b1**2 * self.power.integrals.Plin
            self._kaiser_quad_spline = InterpolatedUnivariateSpline(self.power.integrals.klin, quad_linear)
            return  self._kaiser_quad_spline(k)
    #end _kaiser_quadrupole
    
    #---------------------------------------------------------------------------    
    def _extrapolate_power(self, Pspec, kcut):
        """
        Internal function to do a power law extrapolation of the power spectrum
        at high wavenumbers.
        """
        k = self.power.k
        
        # do a power law extrapolation at high k, if needed
        if k.max() < KMAX:
            
            # compute the power law slope at kcut
            logspline = InterpolatedUnivariateSpline(k, np.log(Pspec))
            slope = derivative(logspline, kcut, dx=1e-3)*kcut
            
            print "power law slope = ", slope
            
            lowk_spline = InterpolatedUnivariateSpline(k, Pspec)
            k_total = np.logspace(np.log10(k.min()), np.log10(KMAX), 1000)
            
            Pspec_total = (k_total <= kcut)*lowk_spline(k) + \
                          (k_total > kcut)*lowk_spline(kcut)*(k_total/kcut)**slope
        return k_total, Pspec_total
    #end _extrapolate_power
    
    #---------------------------------------------------------------------------
    def monopole(self, s, mono_func, smoothing_radius=0., kcut=0.2, linear=False):
        """
        Compute the monopole moment of the configuration space correlation 
        function.
        """
        # compute the minimum wavenumber we need
        kmin = 0.01 / np.amax(s) # integral converges for ks < 0.1
        kmin_model = self.power.k.min() 
        if kmin_model > 0.05: 
            raise ValueError("Power spectrum must be computed down to at least k = 0.05 h/Mpc.")
        
        # first compute the linear correlation values
        linear_contrib = np.zeros(len(s))
        if kmin < kmin_model:
            
            if linear:
                klin = self.power.integrals.klin
                kmax = None
            else:
                klin = np.logspace(kmin, kmin_model, 500)
                kmax = self.power.k.min()
            integrals = _fourier_integrals.Fourier1D(0, kmin, smoothing_radius, 
                                                     klin, self._kaiser_monopole(klin),
                                                     kmax=kmax)
            linear_contrib[:] = integrals.evaluate(s)[:]
            
            if linear:
                return linear_contrib
    
        # now do the contribution from the full model
        model_contrib = np.zeros(len(s))
        
        # do the high k power extrapolation
        self.k_extrap, self.P_extrap = self._extrapolate_power(mono_func(self.power), kcut)
        integrals = _fourier_integrals.Fourier1D(0, kmin_model, smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)
        model_contrib[:] = integrals.evaluate(s)[:]
                                                 
        return linear_contrib + model_contrib
    #end monopole
    
    #---------------------------------------------------------------------------
    # def quadrupole(self, s, quad_func, smoothing_radius=0., kcut=0.2, linear=False):
    #     """
    #     Compute the monopole moment of the configuration space correlation 
    #     function.
    #     """
    #     # compute the minimum wavenumber we need
    #     kmin = 0.01 / np.amax(s)
    #     
    #     # do the extrapolation
    #     if linear:
    #         beta = self.power.f/self.power.b1
    #         quad_linear = (4./3*beta + 4./7*beta**2) * self.power.b1**2 * self.power.integrals.Plin
    #         
    #         self.k_extrap = self.power.integrals.klin
    #         self.P_extrap = quad_linear
    #     else:
    #         self.k_extrap, self.P_extrap = self._extrapolate_power(quad_func(self.power), 
    #                                                                kmin, kcut, 2)
    # 
    #     # initialize the fourier integrals class
    #     integrals = _fourier_integrals.Fourier1D(2, kmin, smoothing_radius, 
    #                                              self.k_extrap, self.P_extrap)
    # 
    #     return -1.*integrals.evaluate(s) 
    # #end quadrupole
    
    #---------------------------------------------------------------------------
    
        
    
