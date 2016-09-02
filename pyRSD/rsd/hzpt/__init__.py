from .._cache import Cache, parameter, cached_property
from ... import numpy as np
from .. import tools
from ...data import hzpt_wiggles

class HaloZeldovichBase(Cache):
    """
    Base class to represent a Halo Zel'dovich power spectrum or 
    correlation function
    """
    def __init__(self, zel, enhance_wiggles=False):
        """
        Parameters
        ----------
        zel : ZeldovichPS, ZeldovichCF
            the Zel'dovich class
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """    
        self.zeldovich = zel
        self.enhance_wiggles = enhance_wiggles
        
        # default base parameters
        self._A0_amp    = 763.74
        self._A0_alpha  = 3.313
        self._R_amp     = 25.252
        self._R_alpha   = 0.297
        self._R1_amp    = 4.132
        self._R1_alpha  = 0.2318
        self._R1h_amp   = 4.8919
        self._R1h_alpha = -0.3831
        self._R2h_amp   = 1.9536
        self._R2h_alpha = 0.2163
        self._W0_alpha  = 1.86
        
    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @parameter
    def enhance_wiggles(self, val):
        """
        Whether to enhance the wiggles
        """
        return val
                
    @parameter
    def sigma8_z(self, val):
        """
        The sigma8 value at z
        """
        if val <= 0.: raise ValueError("`sigma8_z` must be positive")
        self.zeldovich.SetSigma8AtZ(val)
        return val
        
    #---------------------------------------------------------------------------
    # broadband power-law parameters
    #---------------------------------------------------------------------------
    @parameter
    def _A0_amp(self, val):
        return val
        
    @parameter
    def _A0_alpha(self, val):
        return val
        
    @parameter
    def _R_amp(self, val):
        return val
        
    @parameter
    def _R_alpha(self, val):
        return val

    @parameter
    def _R1_amp(self, val):
        return val
        
    @parameter
    def _R1_alpha(self, val):
        return val
        
    @parameter
    def _R1h_amp(self, val):
        return val
        
    @parameter
    def _R1h_alpha(self, val):
        return val
        
    @parameter
    def _R2h_amp(self, val):
        return val
        
    @parameter
    def _R2h_alpha(self, val):
        return val
        
    @parameter
    def _W0_alpha(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------        
    @cached_property()
    def wiggles_data(self):
        """
        Return the HZPT Wiggles+ data
        """
        return hzpt_wiggles()
        
    @cached_property("wiggles_data")
    def wiggles_spline(self):
        """
        Return the HZPT Wiggles+ data
        """
        kw = {'bounds_error':False, 'fill_value':0.}
        return tools.RSDSpline(self.wiggles_data[:,0], self.wiggles_data[:,1], **kw)
    
    #---------------------------------------------------------------------------
    # broadband model parameters
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', '_A0_amp', '_A0_alpha')
    def A0(self):
        """
        Returns the A0 radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are power [(h/Mpc)^3]
        """
        return self._A0_amp*(self.sigma8_z/0.8)**self._A0_alpha

    @cached_property('sigma8_z', '_R_amp', '_R_alpha')
    def R(self):
        """
        Returns the R radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R_amp*(self.sigma8_z/0.8)**self._R_alpha

    @cached_property('sigma8_z', '_R1_amp', '_R1_alpha')
    def R1(self):
        """
        Returns the R1 radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1_amp*(self.sigma8_z/0.8)**self._R1_alpha

    @cached_property('sigma8_z', '_R1h_amp', '_R1h_alpha')
    def R1h(self):
        """
        Returns the R1h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1h_amp*(self.sigma8_z/0.8)**self._R1h_alpha
        
    @cached_property('sigma8_z', '_W0_alpha')
    def W0(self):
        """
        Parameterize the ampltidue of the enhanced wiggles as a function of
        sigma8(z)
        """
        return (self.sigma8_z/0.8)**self._W0_alpha
    
    @cached_property('sigma8_z', '_R2h_amp', '_R2h_alpha')
    def R2h(self):
        """
        Returns the R2h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R2h_amp*(self.sigma8_z/0.8)**self._R2h_alpha

    #---------------------------------------------------------------------------
    # function calls
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Update any attributes with the specified values
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
                
    def __broadband__(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3

        The functional form is given by: 

        P_BB = A0 * F(k) * [ (1 + (k*R1)^2) / (1 + (k*R1h)^2 + (k*R2h)^4) ], 
        as given by Eq. 1 in arXiv:1501.07512.
        """
        F = 1. - 1./(1. + (k*self.R)**2)
        return F*self.A0*(1 + (k*self.R1)**2) / (1 + (k*self.R1h)**2 + (k*self.R2h)**4)

    def __wiggles__(self, k):
        """
        Return the enhanced BAO wiggles if `enhanced_wiggles` is `True`, 
        else just return 0
        """
        return self.W0*self.wiggles_spline(k) if self.enhance_wiggles else 0
        
    @tools.unpacked
    def __zeldovich__(self, k):
        """
        Return the Zel'dovich power at the specified `k`
        """
        return np.nan_to_num(self.zeldovich(k)) # set any NaNs to zero
            
    def __call__(self, k):
        """
        Return the total power at the specified `k`, where the 
        total power is equal to the Zel'dovich power + broadband 
        correction + (optionally) enhanced wiggles
        """
        # make sure sigma8 is set properly
        if self.zeldovich.GetSigma8AtZ() != self.sigma8_z:
            self.zeldovich.SetSigma8AtZ(self.sigma8_z)
            
        return self.__broadband__(k) + self.__zeldovich__(k) + self.__wiggles__(k)
        
        
from .P00 import HaloZeldovichP00, HaloZeldovichCF00
from .P01 import HaloZeldovichP01
from .P11 import HaloZeldovichP11
from .Phm import HaloZeldovichPhm, HaloZeldovichCFhm
from .interpolated import InterpolatedHZPTModels
