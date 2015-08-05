from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from . import tools, INTERP_KMIN, INTERP_KMAX
from .. import numpy as np, pygcl
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
    interpolation_grid['sigma8'] = np.linspace(0.5, 1.5, 100)
    interpolation_grid['k'] = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 200)

    #---------------------------------------------------------------------------
    def __init__(self, z, sigma8, interpolated=False):
        """
        Parameters
        ----------
        z : float
            The desired redshift to compute the power at
        sigma8 : float
            The desired sigma8 to compute the power at
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """
        # initialize the Cache base class
        Cache.__init__(self)
        
        # the model parameters
        self.z            = z
        self.sigma8       = sigma8
        self.interpolated = interpolated
        
        self._A0_amp = 730.
        self._A0_slope = 3.75
        self._R1_amp = 3.25
        self._R1_slope = 0.7
        self._R1h_amp = 3.87
        self._R1h_slope = 0.29
        self._R2h_amp = 1.69
        self._R2h_slope = 0.43           
        
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
    def interpolated(self, val):
        """
        If `True`, return the Zel'dovich power term from an interpolation table
        """
        return val
        
    @parameter
    def z(self, val):
        """
        The redshift
        """
        self.Pzel.SetRedshift(val)
        return val
        
    @parameter
    def sigma8(self, val):
        """
        The sigma8 value
        """
        self.Pzel.SetSigma8(val)
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
    @cached_property('z', 'cosmo')
    def _normalized_sigma8_z(self):
        """
        Return the normalized sigma8(z) from the input cosmology
        """
        return self.cosmo.Sigma8_z(self.z) / self.cosmo.sigma8()
        
    #---------------------------------------------------------------------------
    @cached_property("z")
    def interpolation_table(self):
        """
        Evaluate the Zeldovich power for storing in the interpolation table.
        
        Notes
        -----
        This dependes on the redshift stored in the `z` attribute and must be 
        recomputed whenever that quantity changes.
        """ 
        # the interpolation grid points
        sigma8s = self.interpolation_grid['sigma8']
        ks = self.interpolation_grid['k']
        
        # get the grid values
        grid_vals = []
        for i, s8 in enumerate(sigma8s):
            self.Pzel.SetSigma8(s8)
            grid_vals += list(self.zeldovich_power(ks, ignore_interpolated=True))
        grid_vals = np.array(grid_vals).reshape((len(sigma8s), len(ks)))
        
        # return the interpolator
        return RegularGridInterpolator((sigma8s, ks), grid_vals)

    #---------------------------------------------------------------------------
    @cached_property("sigma8", "_normalized_sigma8_z")
    def sigma8_z(self):
        """
        Return sigma8(z), normalized to the desired sigma8 at z = 0
        """
        return self.sigma8 * self._normalized_sigma8_z
        
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

    #---------------------------------------------------------------------------
    @cached_property('sigma8_z')
    def R(self):
        """
        Returns the R radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 26*(self.sigma8_z/0.8)**0.15

    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', '_R1_amp', '_R1_slope')
    def R1(self):
        """
        Returns the R1 radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1_amp*(self.sigma8_z/0.8)**self._R1_slope

    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', '_R1h_amp', '_R1h_slope')
    def R1h(self):
        """
        Returns the R1h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1h_amp*(self.sigma8_z/0.8)**self._R1h_slope
    
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', '_R2h_amp', '_R2h_slope')
    def R2h(self):
        """
        Returns the R2h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R2h_amp*(self.sigma8_z/0.8)**self._R2h_slope

    #---------------------------------------------------------------------------
    # Function calls
    #---------------------------------------------------------------------------
    def compensation(self, k):
        """
        The compensation function F(k) that causes the broadband power to go
        to zero at low k, in order to conserver mass/momentum

        The functional form is given by 1 - 1 / (1 + k^2 R^2), where R(z) 
        is given by Eq. 4 in arXiv:1501.07512.
        """
        return 1. - 1./(1. + (k*self.R)**2)
    
    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Return the total power, equal to the Zeldovich power + broadband 
        correction
        """
        return self.broadband_power(k) + self.zeldovich_power(k)

    #---------------------------------------------------------------------------
    def broadband_power(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3

        The functional form is given by: 

        P_BB = A0 * F(k) * [ (1 + (k*R1)^2) / (1 + (k*R1h)^2 + (k*R2h)^4) ], 
        as given by Eq. 1 in arXiv:1501.07512.
        """
        F = self.compensation(k)
        return F*self.A0*(1 + (k*self.R1)**2) / (1 + (k*self.R1h)**2 + (k*self.R2h)**4)
    
    #---------------------------------------------------------------------------
    @tools.unpacked
    def zeldovich_power(self, k, ignore_interpolated=False):
        """
        Return the Zel'dovich power at the specified `k`
        """
        if self.interpolated and not ignore_interpolated:
            if np.isscalar(k):
                pts = [self.sigma8, k]
            else:
                pts = np.vstack((np.repeat(self.sigma8, len(k)), k)).T
            return self.interpolation_table(pts)
        else:
            return self.Pzel(k)
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class HaloZeldovichP00(HaloZeldovichPS):
    """
    Halo Zel'dovich P00
    """ 
    def __init__(self, cosmo, z, sigma8, interpolated=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        z : float
            The desired redshift to compute the power at
        sigma8 : float
            The desired sigma8 to compute the power at
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP00(cosmo, z)
        
        # initialize the base class
        super(HaloZeldovichP00, self).__init__(z, sigma8, interpolated)
        
        # save the cosmology too
        self.cosmo = cosmo
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
class HaloZeldovichP01(HaloZeldovichPS):
    """
    Halo Zel'dovich P01
    """ 
    def __init__(self, cosmo, z, sigma8, f, interpolated=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        z : float
            The desired redshift to compute the power at
        sigma8 : float
            The desired sigma8 to compute the power at
        f : float
            The desired logarithmic growth rate
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """  
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP01(cosmo, z)
        
        # initialize the base class
        super(HaloZeldovichP01, self).__init__(z, sigma8, interpolated)
         
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
    
    #---------------------------------------------------------------------------
    @cached_property('f', 'A0')
    def dA0_dlna(self):
        return self.f * 3.75 * self.A0

    #---------------------------------------------------------------------------
    @cached_property('f', 'R')
    def dR_dlna(self):
        return self.f * 0.15 * self.R

    #---------------------------------------------------------------------------
    @cached_property('f', 'R1')
    def dR1_dlna(self):
        return self.f * 0.88 * self.R1

    #---------------------------------------------------------------------------
    @cached_property('f', 'R1h')
    def dR1h_dlna(self):
        return self.f * 0.29 * self.R1h

    #---------------------------------------------------------------------------
    @cached_property('f', 'R2h')
    def dR2h_dlna(self):
        return self.f * 0.43 * self.R2h
            
    #---------------------------------------------------------------------------
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
        if self.Pzel.GetSigma8() != self.sigma8:
            self.Pzel.SetSigma8(self.sigma8)
        
        return self.broadband_power(k) + 2*self.f*self.zeldovich_power(k)
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
