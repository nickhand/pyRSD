"""
 power_gal.py
 pyRSD: subclass of power_biased for a galaxy population
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/09/2014
"""
from . import power_biased
import numpy as np

class GalaxySpectrum(power_biased.BiasedSpectrum):
    
                
    def __init__(self, **kwargs):
        
        # initalize the dark matter power spectrum
        super(GalaxySpectrum, self).__init__(**kwargs)
        
        # don't violate galilean invariance.
        self._include_2loop = False
        
    #end __init__
    
    #---------------------------------------------------------------------------
    # THE GALAXY FRACTIONS
    #---------------------------------------------------------------------------
    @property
    def fs(self):
        """
        The satellite fraction, fs = N_sat / N_gal 
        """
        try:
            return self._fs
        except AttributeError:
            raise ValueError("Must specify satellite fraction 'fs' attribute.")
    
    @fs.setter
    def fs(self, val):
        self._fs = val
        
    #---------------------------------------------------------------------------
    @property
    def fcB(self):
        """
        The centrals with sats (cB) fraction, fcB = N_cB / N_cen 
        """
        try:
            return self._fcB
        except AttributeError:
            raise ValueError("Must specify central with sats fraction 'fcB' attribute.")
    
    @fcB.setter
    def fcB(self, val):
        self._fcB = val
        
    #---------------------------------------------------------------------------
    @property
    def fsB(self):
        """
        The satellite with sats fraction, fsB = N_sB / N_sat
        """
        try:
            return self._fsB
        except AttributeError:
            raise ValueError("Must specify satellite with sats fraction 'fsB' attribute.")
    
    @fsB.setter
    def fsB(self, val):
        self._fsB = val
        
    #---------------------------------------------------------------------------
    # THE GALAXY SAMPLE BIASES
    #---------------------------------------------------------------------------
    @property
    def b1_cA(self):
        """
        The linear bias factor for the centrals with no sats in same halo.
        """
        try:
            return self._b1_cA
        except AttributeError:
            raise ValueError("Must specify cA linear bias 'b1_cA' attribute.")
            
    @b1_cA.setter
    def b1_cA(self, val):
        self._b1_cA = val
    #---------------------------------------------------------------------------
    @property
    def b1_cB(self):
        """
        The linear bias factor for the centrals with sats in same halo.
        """
        try:
            return self._b1_cB
        except AttributeError:
            raise ValueError("Must specify cB linear bias 'b1_cB' attribute.")
            
    @b1_cB.setter
    def b1_cB(self, val):
        self._b1_cB = val
    #---------------------------------------------------------------------------
    @property
    def b1_c(self):
        """
        The linear bias factor for all centrals. This is not a free parameter, 
        but is computed as weighted mean of b1_cA and b1_cB.
        """
        return self.fcB * self.b1_cB + (1. - self.fcB) * self.b1_cA
    #---------------------------------------------------------------------------
    @property
    def b1_sA(self):
        """
        The linear bias factor for satellites with no other sats in same halo.
        """
        try:
            return self._b1_sA
        except AttributeError:
            raise ValueError("Must specify sA linear bias 'b1_sA' attribute.")
            
    @b1_sA.setter
    def b1_sA(self, val):
        self._b1_sA = val
    #---------------------------------------------------------------------------
    @property
    def b1_sB(self):
        """
        The linear bias factor for satellites with other sats in same halo.
        """
        try:
            return self._b1_sB
        except AttributeError:
            raise ValueError("Must specify sB linear bias 'b1_sB' attribute.")
            
    @b1_sB.setter
    def b1_sB(self, val):
        self._b1_sB = val
    #---------------------------------------------------------------------------
    @property
    def b1_s(self):
        """
        The linear bias factor for all satellites. This is not a free parameter, 
        but is computed as weighted mean of b1_sA and b1_sB.
        """
        return self.fsB * self.b1_sB + (1. - self.fsB) * self.b1_sA
    #---------------------------------------------------------------------------
    # VELOCITY DISPERSIONS
    #---------------------------------------------------------------------------
    @property
    def sigma_cs(self):
        """
        The FOG velocity dispersion for central-sats cross spectra Pgal_cAs 
        and Pgal_cBs
        """
        try:
            return self._sigma_cs
        except AttributeError:
            raise ValueError("Must specify velocity dispersion 'sigma_cs' attribute.")
            
    @sigma_cs.setter
    def sigma_cs(self, val):
        self._sigma_cs = val
    #---------------------------------------------------------------------------
    @property
    def sigma_sAsA(self):
        """
        The FOG velocity dispersion for Pgal_sAsA. 
        """
        try:
            return self._sigma_sAsA
        except AttributeError:
            raise ValueError("Must specify velocity dispersion 'sigma_sAsA' attribute.")
            
    @sigma_sAsA.setter
    def sigma_sAsA(self, val):
        self._sigma_sAsA = val
    #---------------------------------------------------------------------------
    @property
    def sigma_sAsB(self):
        """
        The FOG velocity dispersion for Pgal_sAsB. 
        """
        try:
            return self._sigma_sAsB
        except AttributeError:
            raise ValueError("Must specify velocity dispersion 'sigma_sAsB' attribute.")
            
    @sigma_sAsB.setter
    def sigma_sAsB(self, val):
        self._sigma_sAsB = val
    #---------------------------------------------------------------------------
    @property
    def sigma_sBsB_1h(self):
        """
        The FOG velocity dispersion for the 1-halo part of Pgal_sBsB. 
        """
        try:
            return self._sigma_sBsB_1h
        except AttributeError:
            raise ValueError("Must specify velocity dispersion 'sigma_sBsB_1h' attribute.")
            
    @sigma_sBsB_1h.setter
    def sigma_sBsB_1h(self, val):
        self._sigma_sBsB_1h = val
    #---------------------------------------------------------------------------
    @property
    def sigma_sBsB_2h(self):
        """
        The FOG velocity dispersion for the 2-halo part of Pgal_sBsB. 
        """
        try:
            return self._sigma_sBsB_2h
        except AttributeError:
            raise ValueError("Must specify velocity dispersion 'sigma_sBsB_2h' attribute.")
            
    @sigma_sBsB_2h.setter
    def sigma_sBsB_2h(self, val):
        self._sigma_sBsB_2h = val
    #---------------------------------------------------------------------------
    @property
    def fog_model(self):
        """
        Function to return the FOG suppression factor, which reads in a 
        single variable `x = k \mu \sigma`
        """
        try:
            return self._fog_model
        except AttributeError:
            raise ValueError("Must specify the FOG damping model 'fog_model'")
    
    @fog_model.setter
    def fog_model(self, val):
        self._fog_model = val
    #---------------------------------------------------------------------------
    # 1-HALO ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def one_halo_model(self):
        """
        Attribute determining the model for 1-halo terms
        """
        try:
            return self._one_halo_model
        except AttributeError:
            raise ValueError("Must specify 1-halo model 'one_halo_model' attribute.")
            
    @one_halo_model.setter
    def one_halo_model(self, val):
        if not callable(val):
            raise TypeError("Input 1-halo model must a callable function.")
        self._one_halo_model = val
    #---------------------------------------------------------------------------
    @property
    def one_halo_cBs_args(self):
        """
        Any arguments to pass to the one halo model function held in 
        ``self.one_halo_model`` for P_cBs.
        """
        try:
            return self._one_halo_cBs_args
        except AttributeError:
            raise ValueError("Must specify 1-halo args 'one_halo_cBs_args' attribute.")
            
    @one_halo_cBs_args.setter
    def one_halo_cBs_args(self, val):
        self._one_halo_cBs_args = val
    #---------------------------------------------------------------------------
    @property
    def one_halo_sBsB_args(self):
        """
        Any arguments to pass to the one halo model function held in 
        ``self.one_halo_model`` for P_sBsB.
        """
        try:
            return self._one_halo_sBsB_args
        except AttributeError:
            raise ValueError("Must specify 1-halo args 'one_halo_sBsB_args' attribute.")
            
    @one_halo_sBsB_args.setter
    def one_halo_sBsB_args(self, val):
        self._one_halo_sBsB_args = val
    #---------------------------------------------------------------------------
    # POWER SPECTRA TERMS
    #---------------------------------------------------------------------------
    def Pgal_cc(self, mu):
        """
        The central galaxy auto spectrum, assuming no FOG here. This is a 2-halo
        term only.
        """
        # set the linear biases first
        self.b1     = self.b1_c
        self.b1_bar = self.b1_c
        
        # now return the power spectrum here
        return self.power(mu)
    #---------------------------------------------------------------------------
    def Pgal_cAs(self, mu):
        """
        The cross spectrum between centrals with no satellites and satellites. 
        This is a 2-halo term only. 
        """
        # set the linear biases first
        self.b1     = self.b1_cA
        self.b1_bar = self.b1_s
        
        # the FOG damping
        x = self.sigma_cs * mu * self.k
        G = self.fog_model(x)
        
        # now return the power spectrum here
        return G*self.power(mu)
    #---------------------------------------------------------------------------
    def Pgal_sAsA(self, mu):
        """
        The auto spectrum of satellites with no other sats in same halo. This 
        is a 2-halo term only.
        """
        # set the linear biases first
        self.b1     = self.b1_sA
        self.b1_bar = self.b1_sA
        
        # the FOG damping
        x = self.sigma_sAsA * mu * self.k
        G = self.fog_model(x)
        
        # now return the power spectrum here
        return G**2 * self.power(mu)
    #---------------------------------------------------------------------------
    def Pgal_sAsB(self, mu):
        """
        The cross spectrum of satellites with and without no other sats in 
        same halo. This is a 2-halo term only.
        """
        # set the linear biases first
        self.b1     = self.b1_sA
        self.b1_bar = self.b1_sB
        
        # the FOG damping
        x = self.sigma_sAsB * mu * self.k
        G = self.fog_model(x)
        
        # now return the power spectrum here
        return G**2 * self.power(mu)
    #---------------------------------------------------------------------------
    def Pgal_cBs(self, mu):
        """
        The cross spectrum of centrals with sats in the same halo and satellites.
        This has both a 1-halo and 2-halo term only.
        """
        # set the linear biases first
        self.b1     = self.b1_cB
        self.b1_bar = self.b1_s
        
        # the FOG damping
        x = self.sigma_cs * mu * self.k
        G = self.fog_model(x)
        
        # now return the power spectrum here
        return G * (self.power(mu) + self.Pgal_cBs_1h)
    #---------------------------------------------------------------------------
    @property
    def Pgal_cBs_1h(self):
        """
        The 1-halo term for the cross spectrum of centrals with sats in the 
        same halo and satellites.
        """
        return self.one_halo_model(self.k, *self.one_halo_cBs_args)
    #---------------------------------------------------------------------------
    def Pgal_sBsB(self, mu):
        """
        The auto spectrum of satellits with other sats in the same halo.
        This has both a 1-halo and 2-halo term only.
        """
        # set the linear biases first
        self.b1     = self.b1_sB
        self.b1_bar = self.b1_sB
        
        # the FOG damping terms
        x = self.sigma_sBsB_1h * mu * self.k
        G_1h = self.fog_model(x)
        
        x = self.sigma_sBsB_2h * mu * self.k
        G_2h = self.fog_model(x)
        
        # now return the power spectrum here
        return G_2h**2 * self.power(mu) + G_1h**2 * self.Pgal_sBsB_1h
    #---------------------------------------------------------------------------
    @property
    def Pgal_sBsB_1h(self):
        """
        The 1-halo term for the auto spectrum of satellits with other sats 
        in the same halo.
        """
        return self.one_halo_model(self.k, *self.one_halo_sBsB_args)
    #---------------------------------------------------------------------------
    def Pgal_ss(self, mu):
        """
        The total satellite auto spectrum.
        """
        
        return (1. - self.fsB)**2 * self.Pgal_sAsA(mu) + \
                    2*self.fsB*(1-self.fsB)*self.Pgal_sAsB(mu) + \
                    self.fsB**2 * self.Pgal_sBsB(mu) 
    #---------------------------------------------------------------------------
    def Pgal_cs(self, mu):
        """
        The total central-satellite cross spectrum.
        """
        
        return (1. - self.fcB)*self.Pgal_cAs(mu) + self.fcB*self.Pgal_cBs(mu) 
    #---------------------------------------------------------------------------
    def Pgal(self, mu):
        """
        The total redshift-space galaxy power spectrum, combining the individual
        terms
        """
        fss = self.fs**2
        fcs = 2.*self.fs*(1 - self.fs)
        fcc = (1. - self.fs)**2
        
        return fcc * self.Pgal_cc(mu) + fcs * self.Pgal_cs(mu) + fss * self.Pgal_ss(mu)
    #---------------------------------------------------------------------------