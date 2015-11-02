from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from . import tools, INTERP_KMIN, INTERP_KMAX
from .. import numpy as np, pygcl
from ..data import hzpt_wiggles
import itertools

#-------------------------------------------------------------------------------
# Halo Zeldovich base class
#-------------------------------------------------------------------------------
class HaloZeldovichPS(Cache):
    """
    Base class to represent a Halo Zeldovich power spectrum
    """
    # define the interpolation grid for Zel'dovich power
    interpolation_grid = {}
    interpolation_grid['sigma8_z'] = np.linspace(0.3, 1.0, 100)
    interpolation_grid['k'] = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 300)

    #---------------------------------------------------------------------------
    def __init__(self, z, sigma8_z, interpolate=False, enhance_wiggles=False):
        """
        Parameters
        ----------
        z : float
            The desired redshift to compute the power at
        sigma8_z : float
            The desired sigma8 to compute the power at
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """
        # initialize the Cache base class
        Cache.__init__(self)
        
        # the model parameters
        self.z               = z
        self.sigma8_z        = sigma8_z
        self.interpolate     = interpolate
        self.enhance_wiggles = enhance_wiggles
        
        self._A0_amp    = 730.
        self._A0_slope  = 3.75
        self._R1_amp    = 3.25
        self._R1_slope  = 0.7
        self._R1h_amp   = 3.87
        self._R1h_slope = 0.29
        self._R2h_amp   = 1.69
        self._R2h_slope = 0.43  
        self._W0_slope  = 1.86     
        
    def __getstate__(self):
        return self.__dict__
       
    def __setstate__(self, d):
        self.__dict__ = d
        
        # backwards compatibility for HZPTw+
        self.enhance_wiggles = False
        self._W0_slope      = 1.86
       
    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------
    @parameter
    def _A0_amp(self, val):
        return val
        
    @parameter
    def _A0_slope(self, val):
        return val
        
    @parameter
    def _R1_amp(self, val):
        return val
        
    @parameter
    def _R1_slope(self, val):
        return val
        
    @parameter
    def _R1h_amp(self, val):
        return val
        
    @parameter
    def _R1h_slope(self, val):
        return val
        
    @parameter
    def _R2h_amp(self, val):
        return val
        
    @parameter
    def _R2h_slope(self, val):
        return val
        
    @parameter
    def _W0_slope(self, val):
        return val
        
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
    def z(self, val):
        """
        The redshift
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
    # Cached properties
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
            grid_vals += list(self.zeldovich_power(ks, ignore_interpolated=True))
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
    # Model Parameters
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', '_A0_amp', '_A0_slope')
    def A0(self):
        """
        Returns the A0 radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are power [(h/Mpc)^3]
        """
        return self._A0_amp*(self.sigma8_z/0.8)**self._A0_slope

    @cached_property('sigma8_z')
    def R(self):
        """
        Returns the R radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 26*(self.sigma8_z/0.8)**0.15

    @cached_property('sigma8_z', '_R1_amp', '_R1_slope')
    def R1(self):
        """
        Returns the R1 radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1_amp*(self.sigma8_z/0.8)**self._R1_slope

    @cached_property('sigma8_z', '_R1h_amp', '_R1h_slope')
    def R1h(self):
        """
        Returns the R1h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1h_amp*(self.sigma8_z/0.8)**self._R1h_slope
        
    @cached_property('sigma8_z', '_W0_slope')
    def W0(self):
        """
        Parameterize the ampltidue of the enhanced wiggles as a function of
        sigma8(z)
        """
        return (self.sigma8_z/0.8)**self._W0_slope
    
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', '_R2h_amp', '_R2h_slope')
    def R2h(self):
        """
        Returns the R2h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R2h_amp*(self.sigma8_z/0.8)**self._R2h_slope

    #---------------------------------------------------------------------------
    # function calls
    #---------------------------------------------------------------------------
    def compensation(self, k):
        """
        The compensation function F(k) that causes the broadband power to go
        to zero at low k, in order to conserver mass/momentum

        The functional form is given by 1 - 1 / (1 + k^2 R^2), where R(z) 
        is given by Eq. 4 in arXiv:1501.07512.
        """
        return 1. - 1./(1. + (k*self.R)**2)

    def __call__(self, k):
        """
        Return the total power, equal to the Zeldovich power + broadband 
        correction
        """
        # make sure sigma8 is set properly
        if self.Pzel.GetSigma8AtZ() != self.sigma8_z:
            self.Pzel.SetSigma8AtZ(self.sigma8_z)
            
        return self.broadband_power(k) + self.zeldovich_power(k) + self.wiggles_plus(k)

    def wiggles_plus(self, k):
        """
        Return the enhanced BAO wiggles if `enhanced_wiggles` is `True`, 
        else just return 0
        """
        return self.W0*self.wiggles_spline(k) if self.enhance_wiggles else 0
    
    def broadband_power(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3

        The functional form is given by: 

        P_BB = A0 * F(k) * [ (1 + (k*R1)^2) / (1 + (k*R1h)^2 + (k*R2h)^4) ], 
        as given by Eq. 1 in arXiv:1501.07512.
        """
        F = self.compensation(k)
        return F*self.A0*(1 + (k*self.R1)**2) / (1 + (k*self.R1h)**2 + (k*self.R2h)**4)

    @tools.unpacked
    def zeldovich_power(self, k, ignore_interpolated=False):
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
    

#-------------------------------------------------------------------------------
# HZPT P00
#-------------------------------------------------------------------------------
class HaloZeldovichP00(HaloZeldovichPS):
    """
    Halo Zel'dovich P00
    """ 
    def __init__(self, cosmo, z, sigma8_z, **kwargs):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        z : float
            The desired redshift to compute the power at
        sigma8_z : float
            The desired sigma8 to compute the power at
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP00(cosmo, z)
        
        # initialize the base class
        super(HaloZeldovichP00, self).__init__(z, sigma8_z, **kwargs)
        
        # save the cosmology too
        self.cosmo = cosmo
   
   
#-------------------------------------------------------------------------------
# HZPT P01
#-------------------------------------------------------------------------------
class HaloZeldovichP01(HaloZeldovichPS):
    """
    Halo Zel'dovich P01
    """ 
    def __init__(self, cosmo, z, sigma8_z, f, **kwargs):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        z : float
            The desired redshift to compute the power at
        sigma8_z : float
            The desired sigma8 to compute the power at
        f : float
            The desired logarithmic growth rate
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """  
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP01(cosmo, z)
        
        # initialize the base class
        super(HaloZeldovichP01, self).__init__(z, sigma8_z, **kwargs)
         
        # set the parameters
        self.cosmo = cosmo
        self.f = f
        
    #---------------------------------------------------------------------------       
    @parameter
    def f(self, val):
        """
        The logarithmic growth rate
        """
        return val
        
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property('f', 'A0')
    def dA0_dlna(self):
        return self.f * self._A0_slope * self.A0

    @cached_property('f', 'R')
    def dR_dlna(self):
        return self.f * 0.15 * self.R
        
    @cached_property('f', 'R1')
    def dR1_dlna(self):
        return self.f * self._R1_slope * self.R1

    @cached_property('f', 'R1h')
    def dR1h_dlna(self):
        return self.f * self._R1h_slope * self.R1h

    @cached_property('f', 'R2h')
    def dR2h_dlna(self):
        return self.f * self._R2h_slope * self.R2h
        
    @cached_property('f', 'W0')
    def dW0_dlna(self):
        return self.f * self._W0_slope * self.W0
            
    #---------------------------------------------------------------------------
    # main functions
    #---------------------------------------------------------------------------
    def wiggles_plus(self, k):
        """
        Return the enhanced BAO wiggles if `enhanced_wiggles` is `True`, 
        else just return 0
        """
        return self.dW0_dlna*self.wiggles_spline(k) if self.enhance_wiggles else 0
        
    def broadband_power(self, k):
        """
        The broadband power correction for P01 in units of (Mpc/h)^3

        This is basically the derivative of the broadband band term for P00, taken
        with respect to lna
        """
        F = self.compensation(k)

        # store these for convenience
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
        `self.zeldovich_power`
        """
        # make sure sigma8 is set properly
        if self.Pzel.GetSigma8AtZ() != self.sigma8_z:
            self.Pzel.SetSigma8AtZ(self.sigma8_z)
        
        return self.broadband_power(k) + 2*self.f*self.zeldovich_power(k) + self.wiggles_plus(k)
    
    
#-------------------------------------------------------------------------------
class HaloZeldovichPhm(HaloZeldovichPS):
    """
    Halo Zel'dovich Phm
    """ 
    def __init__(self, cosmo, z, sigma8_z, interpolated=False, indep_var='b1'):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        z : float
            The desired redshift to compute the power at
        sigma8_z : float
            The desired sigma8 to compute the power at
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        indep_var : {`b1`, `M`}, optional (`b1`)
            Compute Phm as a function of linear bias b1 or halo mass M
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP00(cosmo, z)
        
        # bias to mass relation
        self.bias_to_mass = tools.BiasToMassRelation(z, cosmo, interpolated)
        
        # initialize the base class
        super(HaloZeldovichPhm, self).__init__(z, sigma8_z, interpolated)
        
        # save the cosmology too
        self.cosmo = cosmo
        self.delta_halo = 200.
        self.indep_var = indep_var
        
    #---------------------------------------------------------------------------
    @parameter
    def delta_halo(self, val):
        """
        The halo bias relation delta halo value to use
        """
        self.bias_to_mass.delta_halo = val
        return val
        
    @parameter
    def indep_var(self, val):
        """
        Either interpolate the model as a function of bias or mass
        """
        if val not in ['b1', 'M']:
            raise ValueError("can only compute model as function of `b1` or `M`")
        return val
        
    @parameter
    def interpolate(self, val):
        """
        If `True`, return the Zel'dovich power term from an interpolation table
        """
        self.bias_to_mass.interpolated = val
        return val
        
    @parameter
    def b1(self, val):
        """
        The linear bias
        """
        return val
    
    @parameter
    def z(self, val):
        """
        The redshift
        """
        self.bias_to_mass.z = val
        return val
        
    #---------------------------------------------------------------------------
    @cached_property('cosmo')
    def rho_bar(self):
        """
        The mean density at z = 0
        """
        return self.cosmo.rho_bar_z(0.)
        
    @cached_property('b1', 'sigma8_z')
    def M(self):
        """
        The mass corresponding to the `b1` attribute in units of `M_sun/h`
        """
        return self.bias_to_mass(self.sigma8_z, self.b1)
        
    @cached_property('sigma8_z', 'b1', 'M', 'indep_var')
    def A0(self):
        """
        Returns the A0 radius parameter

        Note: the units are power [(h/Mpc)^3]
        """
        if self.indep_var == 'M':
            A = 11.462 
            alpha = 0.719
            A_dm = 476.71
            beta = 4.643
            return A*(self.M / self.rho_bar)**alpha + A_dm*(self.sigma8_z/0.8)**beta
        else:
            A = 2035.69288
            alpha = 1.78606
            beta = 3.6492
            return A*(self.b1/2.)**alpha * (self.sigma8_z/0.8)**beta
        
    @cached_property('z', 'sigma8_z', 'b1', 'M', 'indep_var')
    def R1(self):
        """
        Returns the R1 radius parameter

        Note: the units are length [Mpc/h]
        """
        if self.indep_var == 'M':
            alpha = -1.146
            beta = -0.206
            A = 4.837 
            return A * (1 + self.z)**alpha * (self.M/1e13)**beta
        else:
            # alpha = -0.40178
            # beta = -0.64280
            # A = 3.07209
            # return A * (1 + self.z)**alpha * (self.b1/2.)**beta
            alpha = 0.5482
            beta = -0.6438
            A = 2.9813
            return A * (self.sigma8_z/0.8)**alpha * (self.b1/2.)**beta
            
        
    @cached_property('z', 'sigma8_z', 'b1', 'M', 'indep_var')
    def R1h(self):
        """
        Returns the R1h radius parameter

        Note: the units are length [Mpc/h]
        """
        if self.indep_var == 'M':
            alpha = -1.082
            beta = -0.293
            A = 7.242 
            return A * (1 + self.z)**alpha * (self.M/1e13)**beta
        else:
            # alpha = -0.01459
            # beta = -0.88849
            # A = 3.74763
            # return A * (1 + self.z)**alpha * (self.b1/2.)**beta
            alpha = 0.01591489
            beta = -0.88942626
            A = 3.73986252
            return A * (self.sigma8_z/0.8)**alpha * (self.b1/2.)**beta
    
    @cached_property('z', 'sigma8_z', 'b1', 'M', 'indep_var')
    def R2h(self):
        """
        Returns the R2h radius parameter

        Note: the units are length [Mpc/h]
        """
        if self.indep_var == 'M':
            alpha = -1.135
            beta = -0.324
            A = 2.579 
            return A * (1 + self.z)**alpha * (self.M/1e13)**beta
        else:
            # alpha = 0.00188
            # beta = -1.15442
            # A = 1.22377
            # return A * (1 + self.z)**alpha * (self.b1/2.)**beta
            alpha = -0.00818052
            beta = -1.15572913
            A = 1.22260687
            return A * (self.sigma8_z/0.8)**alpha * (self.b1/2.)**beta
        
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
        return self.broadband_power(k) + b1*self.zeldovich_power(k)
    
