from . import HaloZeldovichBase, tools
from . import Cache, parameter, cached_property
from ... import pygcl
        
def default_parameters():
    """
    The default parameters
    
    References
    ----------
    These parameters are from:
    
    file: ``mcmc_fit_kmin-0.005_kmax-1.0.npz``
    directory: ``$RSD_DIR/SimCalibrations/P11HaloZeldovich/results``
    git hash: 8e1304e6
    """
    d = {}
    
    # A0
    d['_A0_amp']   = 658.9
    d['_A0_alpha'] = 3.91
    d['_A0_beta']  = 1.917
    
    # R
    d['_R_amp']   = 18.95
    d['_R_alpha'] = -0.3657
    d['_R_beta']  = -0.2585
            
    # R1h
    d['_R1h_amp']   = 0.8473
    d['_R1h_alpha'] = -0.1524
    d['_R1h_beta']  = 0.7769
    
    return d

class HaloZeldovichP11(HaloZeldovichBase):
    """
    The `mu4` contribution to the radial momentum auto spectrum P11, using HZPT
    
    Notes
    -----
    The 1-loop SPT model for the vector contribution to P11[mu4] should be
    added to the power returned by this class to model the full P11[mu4]
    """ 
    def __init__(self, zel, sigma8_z, f, enhance_wiggles=False):
        """
        Parameters
        ----------
        zel : ZeldovichP11
            the Zel'dovich power spectrum
        sigma8_z : float
            The desired sigma8 to compute the power at
        f : float
            the growth rate
        interpolate : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """   
        super(HaloZeldovichP11, self).__init__(zel, enhance_wiggles)
        self.sigma8_z = sigma8_z
        self.f        = f
        
        # set defaults
        self.update(**default_parameters())
        
    @classmethod
    def from_cosmo(cls, cosmo, sigma8_z, f, enhance_wiggles=False):
        """
        Initialize from a cosmology instance
        """
        Pzel = pygcl.ZeldovichP11(cosmo, 0.)
        return cls(Pzel, sigma8_z, f, enhance_wiggles)
                      
    @parameter
    def f(self, val):
        """
        The logarithmic growth rate
        """
        return val
                
    @parameter
    def _A0_beta(self, val):
        return val
            
    @parameter
    def _R_beta(self, val):
        return val
                
    @parameter
    def _R1h_beta(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # broadband model parameters
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', 'f', '_A0_amp', '_A0_alpha', '_A0_beta')
    def A0(self):
        return self.__powerlaw__(self._A0_amp, self._A0_alpha, self._A0_beta)
        
    @cached_property('sigma8_z', 'f', '_R_amp', '_R_alpha', '_R_beta')
    def R(self):
        return self.__powerlaw__(self._R_amp, self._R_alpha, self._R_beta)

    @cached_property('sigma8_z', 'f', '_R1h_amp', '_R1h_alpha', '_R1h_beta')
    def R1h(self):
        return self.__powerlaw__(self._R1h_amp, self._R1h_alpha, self._R1h_beta)
        
    #---------------------------------------------------------------------------
    # functions
    #---------------------------------------------------------------------------
    def __powerlaw__(self, A, alpha, beta):
        """
        Return a power law as a function of f(z) and sigma8(z) with the specified 
        parameters
        """
        return A * (self.sigma8_z/0.8)**alpha * (self.f/0.5)**beta
        
    def __broadband__(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3
        
        Modeled with a Pade function: A0 / (1 + (k R1h)^2)
        """
        F = 1. - 1./(1. + (k*self.R)**2)
        return F * self.A0 /  (1 + (k*self.R1h)**2)
        
    @tools.unpacked
    def __call__(self, k):
        """
        Return the total power
        """
        if self.zeldovich.GetSigma8AtZ() != self.sigma8_z:
            self.zeldovich.SetSigma8AtZ(self.sigma8_z)
        
        return self.__broadband__(k) + self.f**2 * self.__zeldovich__(k)
