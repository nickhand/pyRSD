from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from . import tools, INTERP_KMIN, INTERP_KMAX
from .. import numpy as np, pygcl
from ..data import hzpt_wiggles

class HaloZeldovichPS(Cache):
    """
    Base class to represent a Halo Zeldovich power spectrum
    """
    # define the interpolation grid for Zel'dovich power
    interpolation_grid = {}
    interpolation_grid['sigma8_z'] = np.linspace(0.3, 1.0, 100)
    interpolation_grid['k'] = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 300)

    def __init__(self, sigma8_z, interpolate=False, enhance_wiggles=False):
        """
        Parameters
        ----------
        sigma8_z : float
            The desired sigma8 to compute the power at
        interpolate : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """
        # initialize the Cache base class
        Cache.__init__(self)
        
        # the model parameters
        self.sigma8_z        = sigma8_z
        self.interpolate     = interpolate
        self.enhance_wiggles = enhance_wiggles
        
        self._A0_amp    = 682.3 #730.
        self._A0_alpha  = 3.85 #3.75
        self._R_amp     = 33.8 #26.0
        self._R_alpha   = 0.049 #0.15
        self._R1_amp    = 2.18 #3.25
        self._R1_alpha  = 0.38 #0.7
        self._R1h_amp   = 2.66 #3.87
        self._R1h_alpha = -0.07 #0.29
        self._R2h_amp   = 1.25 #1.69
        self._R2h_alpha = 0.97 #0.43  
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
    def interpolate(self, val):
        """
        If `True`, return the Zel'dovich power term from an interpolation table
        """
        return val
        
    @parameter
    def sigma8_z(self, val):
        """
        The sigma8 value at z
        """
        self.Pzel.SetSigma8AtZ(val)
        return val
        
    @parameter
    def cosmo(self, val):
        """
        The cosmology of the input linear power spectrum
        """
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
    def interpolation_table(self):
        """
        Evaluate the Zeldovich power for storing in the interpolation table.
        
        Notes
        -----
        This does not depend on redshift, as we are interpolating as a function
        of sigma8(z)
        """ 
        original_s8z = self.sigma8_z
        
        # the interpolation grid points
        sigma8s = self.interpolation_grid['sigma8_z']
        ks = self.interpolation_grid['k']
        
        # get the grid values
        grid_vals = []
        for i, s8 in enumerate(sigma8s):
            self.Pzel.SetSigma8AtZ(s8)
            grid_vals += list(self.__zeldovich__(ks, ignore_interpolated=True))
        grid_vals = np.array(grid_vals).reshape((len(sigma8s), len(ks)))
        
        # return the interpolator
        self.sigma8_z = original_s8z
        return RegularGridInterpolator((sigma8s, ks), grid_vals)
        
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
        for k, v in kwargs.iteritems():
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
    def __zeldovich__(self, k, ignore_interpolated=False):
        """
        Return the Zel'dovich power at the specified `k`
        """
        if self.interpolate and not ignore_interpolated:
            if np.isscalar(k):
                pts = [self.sigma8_z, k]
            else:
                pts = np.vstack((np.repeat(self.sigma8_z, len(k)), k)).T
            return self.interpolation_table(pts)
        else:
            return np.nan_to_num(self.Pzel(k)) # set any NaNs to zero
            
    def __call__(self, k):
        """
        Return the total power at the specified `k`, where the 
        total power is equal to the Zel'dovich power + broadband 
        correction + (optionally) enhanced wiggles
        """
        # make sure sigma8 is set properly
        if self.Pzel.GetSigma8AtZ() != self.sigma8_z:
            self.Pzel.SetSigma8AtZ(self.sigma8_z)
            
        return self.__broadband__(k) + self.__zeldovich__(k) + self.__wiggles__(k)
        
    
class HaloZeldovichP00(HaloZeldovichPS):
    """
    The dark matter auto-spectrum P00 using Halo-Zel'dovich Perturbation Theory
    """ 
    def __init__(self, cosmo, sigma8_z, interpolate=False, enhance_wiggles=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        sigma8_z : float
            The desired sigma8 to compute the power at
        interpolate : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP00(cosmo, 0)
        
        # initialize the base class
        kwargs = {'interpolate':interpolate, 'enhance_wiggles':enhance_wiggles}
        super(HaloZeldovichP00, self).__init__(sigma8_z, **kwargs)
        self.cosmo = cosmo
        
class HaloZeldovichCF00(HaloZeldovichPS):
    """
    The dark matter correlation function using Halo-Zel'dovich Perturbation Theory
    """ 
    def __init__(self, cosmo, sigma8_z):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        sigma8_z : float
            The desired sigma8 to compute the power at
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichCF(cosmo, 0)
        
        # initialize the base class
        kwargs = {'interpolate':False, 'enhance_wiggles':False}
        super(HaloZeldovichCF00, self).__init__(sigma8_z, **kwargs)
        self.cosmo = cosmo
                       
    def __broadband__(self, r):
        """
        The broadband power correction as given by Eq. 7 in arXiv:1501.07512.
        """
        A = -self.A0 * np.exp(-r/self.R) / (4*np.pi*r*self.R**2)
        return A * (1. - (self.R/self.R1h)**2 * np.exp(-r*(self.R + self.R1h)/(self.R*self.R1h)))

    @tools.unpacked
    def __zeldovich__(self, r, ignore_interpolated=False):
        """
        Return the Zel'dovich correlation at the specified `r`
        """
        return np.nan_to_num(self.Pzel(r)) # set any NaNs to zero
            
    def __call__(self, k):
        """
        Return the total correlation
        """
        return self.__broadband__(k) + self.__zeldovich__(k)
   
   
class HaloZeldovichP01(HaloZeldovichPS):
    """
    The dark matter density - radial momentum cross correlation P01 using HZPT
    """ 
    def __init__(self, cosmo, sigma8_z, f, interpolate=False, enhance_wiggles=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        sigma8_z : float
            The desired sigma8 to compute the power at
        f : float
            The desired logarithmic growth rate
        interpolate : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """  
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP01(cosmo, 0.)
        
        # initialize the base class
        kwargs = {'interpolate':interpolate, 'enhance_wiggles':enhance_wiggles}
        super(HaloZeldovichP01, self).__init__(sigma8_z, **kwargs)
         
        # set the parameters
        self.cosmo = cosmo
        self.f = f
    
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
        return self.f * 0.15 * self.R
        
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
        # store these for convenience
        F = 1. - 1./(1. + (k*self.R)**2)
        norm = 1 + (k*self.R1h)**2 + (k*self.R2h)**4
        C = (1. + (k*self.R1)**2) / norm

        # 1st term of tot deriv
        term1 = self.dA0_dlna*F*C;

        # 2nd term
        term2 = self.A0*C * (2*k**2*self.R*self.dR_dlna) / (1 + (k*self.R)**2)**2

        # 3rd term
        term3_a = (2*k**2*self.R1*self.dR1_dlna) / norm
        term3_b = -(1 + (k*self.R1**2)) / norm**2 * (2*k**2*self.R1h*self.dR1h_dlna + 4*k**4*self.R2h**3*self.dR2h_dlna)
        term3 = self.A0*F * (term3_a + term3_b)
        return term1 + term2 + term3
        
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
        if self.Pzel.GetSigma8AtZ() != self.sigma8_z:
            self.Pzel.SetSigma8AtZ(self.sigma8_z)
        
        return self.__broadband__(k) + 2*self.f*self.__zeldovich__(k) + self.__wiggles__(k)
    
    

#-------------------------------------------------------------------------------
# HZPT P11
#-------------------------------------------------------------------------------
class HaloZeldovichP11(HaloZeldovichPS):
    """
    The `mu4` contribution to the radial momentum auto spectrum P11, using HZPT
    
    Notes
    -----
    The 1-loop SPT model for the vector contribution to P11[mu4] should be
    added to the power returned by this class to model the full P11[mu4]
    """ 
    def __init__(self, cosmo, sigma8_z, f, interpolate=False, enhance_wiggles=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        sigma8_z : float
            The desired sigma8 to compute the power at
        f : float
            the growth rate
        interpolate : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP11(cosmo, 0.)
        
        # initialize the base class
        kwargs = {'interpolate':interpolate, 'enhance_wiggles':enhance_wiggles}
        super(HaloZeldovichP11, self).__init__(sigma8_z, **kwargs)
        
        self.cosmo = cosmo
        self.f = f
        
        # A0
        self._A0_amp   = 656
        self._A0_alpha = 3.95
        self._A0_beta  = 1.96
        
        # R
        self._R_amp   = 19.5
        self._R_alpha = -0.60
        self._R_beta  = -0.50
                
        # R1h
        self._R1h_amp   = 0.84
        self._R1h_alpha = -0.045
        self._R1h_beta  = 0.87
                      
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
        if self.Pzel.GetSigma8AtZ() != self.sigma8_z:
            self.Pzel.SetSigma8AtZ(self.sigma8_z)
        
        return self.__broadband__(k) + self.f**2 * self.__zeldovich__(k)
        

class HaloZeldovichPhm(HaloZeldovichPS):
    """
    The halo-matter cross-correlation Phm, using HZPT
    """ 
    def __init__(self, cosmo, sigma8_z, interpolate=False, enhance_wiggles=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        sigma8_z : float
            The desired sigma8 to compute the power at
        interpolate : bool, optional (`False`)
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP00(cosmo, 0.)
                
        # initialize the base class
        kwargs = {'interpolate':interpolate, 'enhance_wiggles':enhance_wiggles}
        super(HaloZeldovichPhm, self).__init__(sigma8_z, **kwargs)
        
        # save the cosmology too
        self.cosmo = cosmo
        
        # A0 power law
        self._A0_amp   = 780.
        self._A0_alpha = 1.57
        self._A0_beta  = 3.50
        
        # R1 power law
        self._R1_amp   = 4.88
        self._R1_alpha = -0.59
        self._R1_beta  = 0.12
        
        # R1h power law
        self._R1h_amp   = 8.00
        self._R1h_alpha = -0.92
        self._R1h_beta  = -0.36
        
        # R2h power law
        self._R2h_amp   = 2.92
        self._R2h_alpha = -1.07
        self._R2h_beta  = -0.35
        
        # R power law
        self._R_amp   = 14.7
        self._R_alpha = 0.22
        self._R_beta  = -0.18
        
        
    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @parameter
    def b1(self, val):
        """
        The linear bias
        """
        return val
    
    #---------------------------------------------------------------------------
    # A0 parameter
    #---------------------------------------------------------------------------
    @parameter
    def _A0_amp(self, val):
        return val
    
    @parameter
    def _A0_alpha(self, val):
        return val
    
    @parameter
    def _A0_beta(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # R parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R_amp(self, val):
        return val
    
    @parameter
    def _R_alpha(self, val):
        return val
    
    @parameter
    def _R_beta(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # R1 parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R1_amp(self, val):
        return val
    
    @parameter
    def _R1_alpha(self, val):
        return val
    
    @parameter
    def _R1_beta(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # R1h parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R1h_amp(self, val):
        return val
    
    @parameter
    def _R1h_alpha(self, val):
        return val
    
    @parameter
    def _R1h_beta(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # R2h parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R2h_amp(self, val):
        return val
    
    @parameter
    def _R2h_alpha(self, val):
        return val
    
    @parameter
    def _R2h_beta(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # cached parameters
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', 'b1', '_A0_amp', '_A0_alpha', '_A0_beta')
    def A0(self):
        """
        Returns the A0 radius parameter

        Note: the units are power [(h/Mpc)^3]
        """
        return self.__powerlaw__(self._A0_amp, self._A0_alpha, self._A0_beta)
        
    @cached_property('sigma8_z', 'b1', '_R1_amp', '_R1_alpha', '_R1_beta')
    def R1(self):
        """
        Returns the R1 radius parameter

        Note: the units are length [Mpc/h]
        """
        return self.__powerlaw__(self._R1_amp, self._R1_alpha, self._R1_beta)
            
    @cached_property('sigma8_z', 'b1', '_R1h_amp', '_R1h_alpha', '_R1h_beta')
    def R1h(self):
        """
        Returns the R1h radius parameter

        Note: the units are length [Mpc/h]
        """
        return self.__powerlaw__(self._R1h_amp, self._R1h_alpha, self._R1h_beta)
    
    @cached_property('sigma8_z', 'b1', '_R2h_amp', '_R2h_alpha', '_R2h_beta')
    def R2h(self):
        """
        Returns the R2h radius parameter

        Note: the units are length [Mpc/h]
        """
        return self.__powerlaw__(self._R2h_amp, self._R2h_alpha, self._R2h_beta)
                
    @cached_property('sigma8_z', 'b1', '_R_amp', '_R_alpha', '_R_beta')
    def R(self):
        """
        Returns the R radius parameter

        Note: the units are length [Mpc/h]
        """
        return self.__powerlaw__(self._R_amp, self._R_alpha, self._R_beta)
        
    #---------------------------------------------------------------------------
    # main functions
    #---------------------------------------------------------------------------        
    def __powerlaw__(self, A, alpha, beta):
        """
        Return a power law as a linear bias and sigma8(z) with the specified 
        parameters
        """
        return A * self.b1**alpha * (self.sigma8_z/0.8)**beta
        
    def __call__(self, b1, k):
        """
        Return the total power, equal to the b1 * Zeldovich power + broadband 
        correction
        
        Parameters
        ----------
        b1 : float
            the linear bias to compute the bias at
        k : array_like
            the wavenumbers in `h/Mpc` to compute the power at
        """
        # make sure sigma8 is set properly
        if self.Pzel.GetSigma8AtZ() != self.sigma8_z:
            self.Pzel.SetSigma8AtZ(self.sigma8_z)
            
        self.b1 = b1
        return self.__broadband__(k) + b1*self.__zeldovich__(k) + b1*self.__wiggles__(k)
