"""
 power_halo.py
 pyRSD: subclass of power_biased.BiasedSpectrum for halos
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/09/2014
"""
from ._cache import cached_property
from .power_biased import BiasedSpectrum

class HaloSpectrum(BiasedSpectrum):
    
                
    def __init__(self, **kwargs):
        
        # initalize the dark matter power spectrum
        super(HaloSpectrum, self).__init__(**kwargs)

    #---------------------------------------------------------------------------
    @cached_property("b1")
    def b1_bar(self):
        """
        The linear bias factor.
        """
        return self.b1

    #---------------------------------------------------------------------------
    @cached_property("b2_00")
    def b2_00_bar(self):
        """
        The quadratic, local bias used for the P00_ss term.
        """
        return self.b2_00
        
    #---------------------------------------------------------------------------
    @cached_property("b2_01")
    def b2_01_bar(self):
        """
        The quadratic, local bias used for the P01_ss term.
        """
        return self.b2_01
            
    #---------------------------------------------------------------------------
    @cached_property("bs")
    def bs_bar(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        return self.bs
    #---------------------------------------------------------------------------