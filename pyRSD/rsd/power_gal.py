"""
 power_gal.py
 pyRSD: subclass of power_biased for a galaxy population
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/09/2014
"""
from . import power_biased, tools
from .. import numpy as np

class GalaxySpectrum(power_biased.BiasedSpectrum):
    
    allowable_kwargs = power_biased.BiasedSpectrum.allowable_kwargs + ['use_mean_bias']
    
    def __init__(self, use_mean_bias=False, **kwargs):
        
        # initalize the dark matter power spectrum
        super(GalaxySpectrum, self).__init__(**kwargs)
        
        # don't violate galilean invariance.
        self._include_2loop = False
        
        self.use_mean_bias = use_mean_bias
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
    def sigma_c(self):
        """
        The FOG velocity dispersion for centrals
        """
        try:
            return self._sigma_c
        except AttributeError:
            raise ValueError("Must specify velocity dispersion 'sigma_c' attribute.")
           
            
    @sigma_c.setter
    def sigma_c(self, val):
        self._sigma_c = val
    
    #---------------------------------------------------------------------------
    @property
    def sigma_s(self):
        """
        The FOG velocity dispersion for satellites
        """
        try:
            return self._sigma_s
        except AttributeError:
            return tools.sigma_from_bias(self.b1_s, self.z, self.power_lin)
            
    @sigma_s.setter
    def sigma_s(self, val):
        self._sigma_s = val
        
    @sigma_s.deleter
    def sigma_s(self, val):
        if hasattr(self, '_sigma_s'): delattr(self, '_sigma_s')
        
    #---------------------------------------------------------------------------
    @property
    def sigma_sA(self):
        """
        The FOG velocity dispersion for "type A" satellites
        """
        try:
            return self._sigma_sA
        except AttributeError:
            return tools.sigma_from_bias(self.b1_sA, self.z, self.power_lin)
            
    @sigma_sA.setter
    def sigma_sA(self, val):
        self._sigma_sA = val
        
    @sigma_sA.deleter
    def sigma_sA(self, val):
        if hasattr(self, '_sigma_sA'): delattr(self, '_sigma_sA')   
        
    #---------------------------------------------------------------------------
    @property
    def sigma_sB(self):
        """
        The FOG velocity dispersion for "type B" satellites
        """
        try:
            return self._sigma_sB
        except AttributeError:
            return tools.sigma_from_bias(self.b1_sB, self.z, self.power_lin)
            
    @sigma_sB.setter
    def sigma_sB(self, val):
        self._sigma_sB = val
    
    @sigma_sB.deleter
    def sigma_sB(self, val):
        if hasattr(self, '_sigma_sB'): delattr(self, '_sigma_sB')
        
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
    @tools.mu_vectorize
    def Pgal_cc(self, mu):
        """
        The central galaxy auto spectrum, assuming no FOG here. This is a 2-halo
        term only.
        """
        # set the linear biases first
        self.b1     = self.b1_c
        self.b1_bar = self.b1_c
           
        # FOG damping
        G = self.fog_model(self.sigma_c*mu*self.k)
        
        # now return the power spectrum here
        return G**2 * self.power(mu)
        
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_cAs(self, mu):
        """
        The cross spectrum between centrals with no satellites and satellites. 
        This is a 2-halo term only. 
        """
        # set the linear biases first
        if self.use_mean_bias:
            mean_bias = np.sqrt(self.b1_cA*self.b1_s)
            self.b1 = self.b1_bar = mean_bias
        else:
            self.b1     = self.b1_cA
            self.b1_bar = self.b1_s
        
        # the FOG damping
        G_c = self.fog_model(self.sigma_c*mu*self.k)
        G_s = self.fog_model(self.sigma_s*mu*self.k)
        
        # now return the power spectrum here
        return G_c*G_s*self.power(mu)
        
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_sAsA(self, mu):
        """
        The auto spectrum of satellites with no other sats in same halo. This 
        is a 2-halo term only.
        """
        # set the linear biases first
        self.b1     = self.b1_sA
        self.b1_bar = self.b1_sA
        
        # the FOG damping
        G = self.fog_model(self.sigma_sA*mu*self.k)
        
        # now return the power spectrum here
        return G**2 * self.power(mu)
        
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_sAsB(self, mu):
        """
        The cross spectrum of satellites with and without no other sats in 
        same halo. This is a 2-halo term only.
        """
        # set the linear biases first
        if self.use_mean_bias:
            mean_bias = np.sqrt(self.b1_sA*self.b1_sB)
            self.b1 = self.b1_bar = mean_bias
        else:
            self.b1     = self.b1_sA
            self.b1_bar = self.b1_sB
        
        # the FOG damping
        G_sA = self.fog_model(self.sigma_sA*mu*self.k)
        G_sB = self.fog_model(self.sigma_sB*mu*self.k)
        
        # now return the power spectrum here
        return G_sA*G_sB*self.power(mu)
    
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_cBs(self, mu):
        """
        The cross spectrum of centrals with sats in the same halo and satellites.
        This has both a 1-halo and 2-halo term only.
        """
        return self.Pgal_cBs_2h(mu) + self.Pgal_cBs_1h(mu)
    
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_cBs_2h(self, mu):
        """
        The 2-halo term for the cross spectrum of centrals with sats in 
        the same halo and satellites.
        """
        # set the linear biases first
        if self.use_mean_bias:
            mean_bias = np.sqrt(self.b1_cB*self.b1_s)
            self.b1 = self.b1_bar = mean_bias
        else:
            self.b1     = self.b1_cB
            self.b1_bar = self.b1_s
        
        # the FOG damping
        G_c = self.fog_model(self.sigma_c*mu*self.k)
        G_s = self.fog_model(self.sigma_s*mu*self.k)
        
        # now return the power spectrum here
        return G_c*G_s*self.power(mu)
    
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_cBs_1h(self, mu):
        """
        The 1-halo term for the cross spectrum of centrals with sats in the 
        same halo and satellites, with mu dependence introduced by the 
        FOG damping
        """
        # the FOG damping
        G_c = self.fog_model(self.sigma_c*mu*self.k)
        G_s = self.fog_model(self.sigma_s*mu*self.k)
        
        return G_c*G_s * self.one_halo_model(self.k, *self.one_halo_cBs_args)

    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_sBsB(self, mu):
        """
        The auto spectrum of satellits with other sats in the same halo.
        This has both a 1-halo and 2-halo term only.
        """
        return self.Pgal_sBsB_2h(mu) + self.Pgal_sBsB_1h(mu)
    
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_sBsB_2h(self, mu):
        """
        The 2-halo term for the auto spectrum of satellits with other sats 
        in the same halo.
        """
        # set the linear biases first
        self.b1     = self.b1_sB
        self.b1_bar = self.b1_sB
        
        # the FOG damping terms
        G = self.fog_model(self.sigma_sB*mu*self.k)
        
        # now return the power spectrum here
        return G**2 * self.power(mu)
        
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_sBsB_1h(self, mu):
        """
        The 1-halo term for the auto spectrum of satellits with other sats 
        in the same halo, with mu dependence introduced by the FOG damping
        """
        G = self.fog_model(self.sigma_sB*mu*self.k)
        return G**2 * self.one_halo_model(self.k, *self.one_halo_sBsB_args)
        
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_ss(self, mu):
        """
        The 2-halo part of the total satellite auto spectrum.
        """
        
        return (1. - self.fsB)**2 * self.Pgal_sAsA(mu) + \
                    2*self.fsB*(1-self.fsB)*self.Pgal_sAsB(mu) + \
                    self.fsB**2 * self.Pgal_sBsB(mu) 
                    
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
    def Pgal_cs(self, mu):
        """
        The total central-satellite cross spectrum.
        """
        
        return (1. - self.fcB)*self.Pgal_cAs(mu) + self.fcB*self.Pgal_cBs(mu) 
    #---------------------------------------------------------------------------
    @tools.mu_vectorize
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
    @tools.monopole
    def Pgal_mono(self, mu):
        """
        The total redshift-space galaxy monopole moment
        """
        return self.Pgal(mu)
    #---------------------------------------------------------------------------
    @tools.quadrupole
    def Pgal_quad(self, mu):
        """
        The total redshift-space galaxy quadrupole moment
        """
        return self.Pgal(mu)
    #---------------------------------------------------------------------------
    @tools.hexadecapole
    def Pgal_hexadec(self, mu):
        """
        The total redshift-space galaxy hexadecapole moment
        """
        return self.Pgal(mu)
    #---------------------------------------------------------------------------
    