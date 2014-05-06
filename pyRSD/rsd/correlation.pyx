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
    def _extrapolate_power(self, Pspec, kmin, kcut, multipole):
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
            if self.power.integrals.klin.min() > kmin:
                raise ValueError("Minimum wavenumber needed for convergence too low.")
                
                
            # compute the linear monopole or quadrupole
            if multipole == 0:
                beta = self.power.f/self.power.b1
                linear_power = (1. + 2./3*beta + 1/5*beta**2) * self.power.b1**2 * self.power.integrals.Plin
            elif multipole == 2:
                beta = self.power.f/self.power.b1
                linear_power = (4./3*beta + 4./7*beta**2) * self.power.b1**2 * self.power.integrals.Plin
                    
            lowk_interp = _functionator.splineInterpolator(self.power.integrals.klin, linear_power)
            lowk        = np.logspace(np.log10(kmin), np.log10(k.min()), 200)[:-1]
            lowP        = lowk_interp(lowk)
            
            k     = np.concatenate( (lowk, k) )
            Pspec = np.concatenate( (lowP, Pspec) )
        
        # also, do a power law extrapolation at high k, if needed
        if k.max() < KMAX:
            
            # use the largest k value as the point of extrapolation
            one_sided = False
            if kcut is None or kcut > k.max():
                kcut = k[-1]
                one_sided = True
                
            if not one_sided:
                kcut_min = 0.9*kcut
                kcut_max = kcut
            else:
                kcut_min = 0.95*kcut
                kcut_max = 1.05*kcut
                
            inds = np.where((k >= kcut_min)*(k <= kcut_max))
            if len(inds[0]) < 5:
                raise ValueError("Not enough points to fit power law extrapolation "
                                 "at high k using k = [%.3f, %.3f]" %(kcut_min, kcut_max))
                                 
            
            p_fit = np.polyfit(np.log(k[inds]), np.log(Pspec[inds]), 1)
            p_gamma, p_amp = p_fit[0], np.exp(p_fit[1])
            powerlaw_extrap = _functionator.powerLawExtrapolator(gamma=p_gamma, A=p_amp)
            
            print "power law slope = ", p_gamma
            
            imin = (np.abs(Pspec[inds] - powerlaw_extrap(k[inds]))).argmin()
            k0   = k[inds][imin]
            P0   = Pspec[inds][imin]
            inds = np.where(k < k0)
            
            print "joining functions at k = ", k0
            print "difference in power here = ", abs(P0 - powerlaw_extrap(k0))/P0
            
            k_extrap = np.linspace(k0, KMAX, 200)
            k        = np.concatenate( (k[inds], k_extrap) )
            Pspec    = np.concatenate( (Pspec[inds], P0*(k_extrap/k0)**p_gamma) )
        
        return k, Pspec
    #end _extrapolate_power
    
    #---------------------------------------------------------------------------
    def monopole(self, s, mono_func, smoothing_radius=0., kcut=0.2, linear=False):
        """
        Compute the monopole moment of the configuration space correlation 
        function.
        """
        # compute the minimum wavenumber we need
        kmin = 0.1 / np.amax(s) # integral converges for ks < 0.1
        
        # do the power extrapolation
        if linear:
            beta = self.power.f/self.power.b1
            mono_linear = (1. + 2./3*beta + 1/5*beta**2) * self.power.b1**2 * self.power.integrals.Plin
            
            self.k_extrap = self.power.integrals.klin
            self.P_extrap = mono_linear
        else:
            self.k_extrap, self.P_extrap = self._extrapolate_power(mono_func(self.power), 
                                                                   kmin, kcut, 0)
        
        # initialize the fourier integrals class
        integrals = _fourier_integrals.Fourier1D(0, kmin, smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)
                                                 
        return integrals.evaluate(s) 
    #end monopole
    
    #---------------------------------------------------------------------------
    def quadrupole(self, s, quad_func, smoothing_radius=0., kcut=0.2, linear=False):
        """
        Compute the monopole moment of the configuration space correlation 
        function.
        """
        # compute the minimum wavenumber we need
        kmin = 0.1 / np.amax(s)
        
        # do the extrapolation
        if linear:
            beta = self.power.f/self.power.b1
            quad_linear = (4./3*beta + 4./7*beta**2) * self.power.b1**2 * self.power.integrals.Plin
            
            self.k_extrap = self.power.integrals.klin
            self.P_extrap = quad_linear
        else:
            self.k_extrap, self.P_extrap = self._extrapolate_power(quad_func(self.power), 
                                                                   kmin, kcut, 2)

        # initialize the fourier integrals class
        integrals = _fourier_integrals.Fourier1D(2, kmin, smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)

        return -1.*integrals.evaluate(s) 
    #end quadrupole
    
    #---------------------------------------------------------------------------
    
        
    
