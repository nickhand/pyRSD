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
    @property
    def b1(self):
        """
        The linear bias factor.
        """
        return self.power.b1
            
    @b1.setter
    def b1(self, val):
        self.power.update(b1=val)
                        
        # delete the linear kaiser splines
        for a in ['_kaiser_mono_spline', '_kaiser_quad_spline']:
            if a in self.__dict__: del self.__dict__[a]
        
    #---------------------------------------------------------------------------
    def kaiser_monopole(self, k):
        """
        The linear, Kaiser power spectrum monopole, given by:
        
        .. math:: P_0(k) = (b_1^2 + 2/3*f*b_1 + 1/5*f^2) * P_{lin}
        
        Parameters
        ----------
        k : array_like or float
            The wavenumber in h/Mpc to compute the monopole at.
        """
        try:
            return self._kaiser_mono_spline(k)
        except:
            beta = self.power.f/self.b1
            mono_linear = (1. + 2./3*beta + 1/5*beta**2) * self.b1**2 * self.power.integrals.Plin
            self._kaiser_mono_spline = InterpolatedUnivariateSpline(self.power.integrals.klin, mono_linear)
            return  self._kaiser_mono_spline(k)
    #end _kaiser_monopole
    
    #---------------------------------------------------------------------------
    def kaiser_quadrupole(self, k):
        """
        The linear, Kaiser power spectrum quadrupole, given by:
        
        .. math:: P_0(k) = (4/3*f*b_1 + 4/7*f^2) * P_{lin}
        
        Parameters
        ----------
        k : array_like or float
            The wavenumber in h/Mpc to compute the quadrupole at.
        """
        try:
            return self._kaiser_quad_spline(k)
        except:
            beta = self.power.f/self.b1
            quad_linear = (4./3*beta + 4./7*beta**2) * self.b1**2 * self.power.integrals.Plin
            self._kaiser_quad_spline = InterpolatedUnivariateSpline(self.power.integrals.klin, quad_linear)
            return  self._kaiser_quad_spline(k)
    #end _kaiser_quadrupole
    
    #---------------------------------------------------------------------------    
    def _extrapolate_power(self, Pspec, kcut, kmin, multipole):
        """
        Internal function to do a power law extrapolation of the power spectrum
        at high wavenumbers.
        """
        k = self.power.k
        kmin_model = k.min()
        
        # do a linear extrapolation at low k
        lowk_extrap = False
        lowk_func = lambda k: k*0.
        if kmin_model > kmin:
            
            if kmin_model > 0.05: 
                raise ValueError("Power spectrum must be computed down to at least k = 0.05 h/Mpc.")
            
            lowk_extrap = True
            if multipole == 0:
                
                # add in the stochasticity if it exists
                if hasattr(self.power, 'stochasticity'):
                    stoch_spline = InterpolatedUnivariateSpline(k, self.power.stochasticity)
                    lowk_func = lambda k: self.kaiser_monopole(k) + stoch_spline(k)
                else:
                    lowk_func = lambda k: self.kaiser_monopole(k)
            elif multipole == 2:
                lowk_func = lambda k: self.kaiser_quadrupole(k)
                
        # do a power law extrapolation at high k, if needed
        highk_extrap = False
        kmax_model = k.max()
        slope = 0.
        if k.max() < KMAX:
            
            # compute the power law slope at kcut
            inds = np.where(Pspec > 0.)
            if k[inds].max() < kcut:
                raise ValueError("Power spectrum is negative at k = kcut.")
            
            highk_extrap = True
            kmax_model = kcut
            logspline = InterpolatedUnivariateSpline(k[inds], np.log(Pspec[inds]))
            slope = derivative(logspline, kcut, dx=1e-3)*kcut
            
        # now compute the combined model
        model_spline = InterpolatedUnivariateSpline(k, Pspec)
        k_total = np.logspace(np.log10(kmin), np.log10(KMAX), 1000)
        
        # the three pieces
        Pspec_low = (lowk_extrap)*(k_total < kmin_model)*lowk_func(k_total)
        Pspec_middle = (k_total <= kmax_model)*(k_total >= kmin_model)*model_spline(k_total)
        Pspec_high = (highk_extrap)*(k_total > kmax_model)*model_spline(kcut)*(k_total/kcut)**slope
        
        Pspec_total = Pspec_low + Pspec_middle + Pspec_high
                          
        return k_total, Pspec_total
    #end _extrapolate_power
    
    #---------------------------------------------------------------------------
    def monopole(self, s, smoothing_radius=0., kcut=0.2, linear=False):
        """
        Compute the correlation function monopole in configuration space by 
        Fourier transforming the power spectrum stored in ``self.power``.
        
        Parameters
        ----------
        s : array_like or float
            The configuration space separations to compute the correlation monopole
            at [units :math: `Mpc h^{-1}`].
        smoothing_radius : float, optional
            The smoothing radius R of the Gaussian filter to apply in Fourer
            space, :math: `exp[-(kR)^2]`. Default is ``smoothing_radius = 0``.
        kcut : float, optional
            The wavenumber above which a power law extrapolation is used for 
            the model in Fourier space. [units :math: `h Mpc^{-1}`]. Default
            is ``kcut = 0.2``.
        linear : bool, optional
            If ``True``, return the linear correlation monopole. 
            
        Returns
        -------
        xi0 : array_like or float
            The correlation function monopole defined at s. [units: dimensionless]
        """
        # compute the minimum wavenumber we need
        # integral should converge for ks < 0.01
        kmin = 0.01 / np.amax(s)
        
        # check if we want the linear correlation
        if linear:
            klin = self.power.integrals.klin
            integrals = _fourier_integrals.Fourier1D(0, kmin, smoothing_radius, 
                                                     klin, self.kaiser_monopole(klin))
            xi0 = integrals.evaluate(s)
            return xi0
    
        # do the high k power extrapolation
        self.k_extrap, self.P_extrap = self._extrapolate_power(self.power.monopole(linear=False), 
                                                               kcut, kmin, 0)
        
        # compute the FT of the model
        integrals = _fourier_integrals.Fourier1D(0, kmin, smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)
        xi0 = integrals.evaluate(s)
        return xi0
    #end monopole
    
    #---------------------------------------------------------------------------
    def quadrupole(self, s, smoothing_radius=0., kcut=0.2, linear=False):
        """
        Compute the correlation function quadrupole in configuration space by 
        Fourier transforming the power spectrum stored in ``self.power``.
        
        Parameters
        ----------
        s : array_like or float
            The configuration space separations to compute the correlation monopole
            at [units :math: `Mpc h^{-1}`]..
        smoothing_radius : float, optional
            The smoothing radius R of the Gaussian filter to apply in Fourer
            space, :math: `exp[-(kR)^2]`. Default is ``smoothing_radius = 0``.
        kcut : float, optional
            The wavenumber above which a power law extrapolation is used for 
            the model in Fourier space. [units :math: `h Mpc^{-1}`]. Default
            is ``kcut = 0.2``.
        linear : bool, optional
            If ``True``, return the linear correlation monopole. 
            
        Returns
        -------
        xi2 : array_like or float
            The correlation function quadrupole defined at s. [units: dimensionless]
        """
        # compute the minimum wavenumber we need
        # integral should converge for ks < 0.01
        kmin = 0.01 / np.amax(s)
        
        # check if we want the linear correlation
        if linear:
            klin = self.power.integrals.klin
            integrals = _fourier_integrals.Fourier1D(2, kmin, smoothing_radius, 
                                                     klin, self.kaiser_quadrupole(klin))
            xi2 = -1.*integrals.evaluate(s)
            return xi2
    
        # do the high k power extrapolation
        self.k_extrap, self.P_extrap = self._extrapolate_power(self.power.quadrupole(linear=False), 
                                                               kcut, kmin, 2)
        
        # compute the FT of the model
        integrals = _fourier_integrals.Fourier1D(2, kmin, smoothing_radius, 
                                                 self.k_extrap, self.P_extrap)
        xi2 = -1.*integrals.evaluate(s)
        return xi2
    #end quadrupole
    
    #---------------------------------------------------------------------------
    
        
    
