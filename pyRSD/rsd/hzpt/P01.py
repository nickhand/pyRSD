from . import HaloZeldovichBase, tools
from . import Cache, parameter, cached_property
from ... import pygcl
        
class HaloZeldovichP01(HaloZeldovichBase):
    """
    The dark matter density - radial momentum cross correlation P01 using HZPT
    """ 
    def __init__(self, zel, sigma8_z, f, enhance_wiggles=False):
        """
        Parameters
        ----------
        zel : ZeldovichP01
            the Zel'dovich power spectrum
        sigma8_z : float
            The desired sigma8 to compute the power at
        f : float
            The desired logarithmic growth rate
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """  
        super(HaloZeldovichP01, self).__init__(zel, enhance_wiggles)
        self.sigma8_z = sigma8_z
        self.f        = f
        
    @classmethod
    def from_cosmo(cls, cosmo, sigma8_z, f, enhance_wiggles=False):
        """
        Initialize from a cosmology instance
        """
        Pzel = pygcl.ZeldovichP01(cosmo, 0.)
        return cls(Pzel, sigma8_z, f, enhance_wiggles)
    
    @parameter
    def f(self, val):
        """
        The logarithmic growth rate
        """
        return val
        
    #---------------------------------------------------------------------------
    # derivative parameters for broadband power
    #---------------------------------------------------------------------------
    @cached_property('f', 'A0')
    def dA0_dlna(self):
        return self.f * self._A0_alpha * self.A0

    @cached_property('f', 'R')
    def dR_dlna(self):
        return self.f * self._R_alpha * self.R
        
    @cached_property('f', 'R1')
    def dR1_dlna(self):
        return self.f * self._R1_alpha * self.R1

    @cached_property('f', 'R1h')
    def dR1h_dlna(self):
        return self.f * self._R1h_alpha * self.R1h

    @cached_property('f', 'R2h')
    def dR2h_dlna(self):
        return self.f * self._R2h_alpha * self.R2h
        
    @cached_property('f', 'W0')
    def dW0_dlna(self):
        return self.f * self._W0_alpha * self.W0
            
    #---------------------------------------------------------------------------
    # main functions
    #---------------------------------------------------------------------------
    def __wiggles__(self, k):
        """
        Return the enhanced BAO wiggles if `enhanced_wiggles` is `True`, 
        else just return 0
        """
        return self.dW0_dlna*self.wiggles_spline(k) if self.enhance_wiggles else 0
        
    def __broadband__(self, k):
        """
        The broadband power correction for P01 in units of (Mpc/h)^3

        This is the derivative of the broadband band term for P00, taken
        with respect to ``lna``
        """        
        k2 = k**2; k4 = k2**2
        A0, R, R1, R1h, R2h = self.A0, self.R, self.R1, self.R1h, self.R2h
        dA0, dR, dR1, dR1h, dR2h = self.dA0_dlna, self.dR_dlna, self.dR1_dlna, self.dR1h_dlna, self.dR2h_dlna

        N = 1 + k2*R**2
        F = 1 - 1/N
        numer = 1 + k2*R1**2
        denom = 1 + k2*R1h**2 + k4*R2h**4

        # derivative wrt A0
        dPbb_dA0 = F*numer/denom * dA0
        
        # derivative wrt R
        dPbb_dR = 2*A0*k2*R * numer / N**2 / denom * dR
        
        # derivative wrt R1
        dPbb_dR1 = 2*A0*k4*R**2*R1 / N / denom * dR1
        
        # derivative wrt R1h
        dPbb_dR1h = -2*A0*k4*R**2*numer*R1h / N / denom**2 * dR1h
        
        # derivative wrt R2h
        dPbb_dR2h = -4*A0*k**6*R**2*numer*R2h**3 / N / denom**2 * dR2h

        return dPbb_dA0 + dPbb_dR + dPbb_dR1 + dPbb_dR1h + dPbb_dR2h
        
    @tools.unpacked
    def __call__(self, k):
        """
        Return the full Halo Zeldovich P01, optionally using the interpolation
        table to compute the Zeldovich part
        
        Note
        ----
        The true Zel'dovich term is 2*f times the result returned by
        `self.__zeldovich__`
        """
        # make sure sigma8 is set properly
        if self.zeldovich.GetSigma8AtZ() != self.sigma8_z:
            self.zeldovich.SetSigma8AtZ(self.sigma8_z)
        
        return self.__broadband__(k) + 2*self.f*self.__zeldovich__(k) + self.__wiggles__(k)
