from .. import pygcl, numpy as np
from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy.optimize import brentq

import warnings
import functools
import itertools

warnings.filterwarnings("ignore", category=DeprecationWarning,module="scipy")

#-------------------------------------------------------------------------------
# DECORATORS
#-------------------------------------------------------------------------------  
def unpacked(method):
    """
    Decorator to avoid return lists/tuples of length 1
    """
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return result if len(result) != 1 else result[0]
        except:
            return result
    return _decorator

#-------------------------------------------------------------------------------
def monopole(f):
    """
    Decorator to compute the monopole from a `self.power` function
    """     
    def wrapper(self, *args, **kwargs):
        k = args[0]
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, k, mus, **kwargs)
        return np.array([simps(pk, x=mus) for pk in Pkmus])
    return wrapper

#-------------------------------------------------------------------------------
def quadrupole(f):
    """
    Decorator to compute the quadrupole from a `self.power` function
    """     
    def wrapper(self, *args, **kwargs):
        k = args[0]
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, k, mus, **kwargs)
        kern = 2.5*(3*mus**2 - 1.)
        return np.array([simps(kern*pk, x=mus) for pk in Pkmus])
    return wrapper

#-------------------------------------------------------------------------------
def hexadecapole(f):
    """
    Decorator to compute the hexadecapole from a `self.power` function
    """  
    def wrapper(self, *args, **kwargs):
        k = args[0]
        mus = np.linspace(0., 1., 1001)
        Pkmus = f(self, k, mus, **kwargs)
        kern = 9./8.*(35*mus**4 - 30.*mus**2 + 3.)
        return np.array([simps(kern*pk, x=mus) for pk in Pkmus])
    return wrapper
    

#-------------------------------------------------------------------------------
# InterpolatedUnivariateSpline with extrapolation
#-------------------------------------------------------------------------------
class RSDSpline(InterpolatedUnivariateSpline):
    """
    Class to implement an `InterpolatedUnivariateSpline` that remembers 
    the x-domain
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        x : (N,) array_like
            Input dimension of domain points -- must be increasing
        y : (N,) array_like
            input dimension of data points
        bounds_error : bool, optional
            If `True`, raise an exception if the desired input domain value
            is out of the input range. Default is `False`
        fill_value : float, optional
            The fill value to use for any values that are out of bounds 
            on the input domain. Default is `0`
        extrap : bool, optional
            If desired domain value is out of bounds, do a linear extrapolation.
            Default is `False`
        """
        
        # default kwargs
        self.bounds_error = kwargs.pop('bounds_error', False)
        self.fill_value   = kwargs.pop('fill_value', 0.)
        self.extrap       = kwargs.pop('extrap', False)
        
        self.x = args[0]
        self.y = args[1]
        super(RSDSpline, self).__init__(*args, **kwargs)

    #----------------------------------------------------------------------------
    def _check_bounds(self, x_new):
        """
        Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """
        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x.  Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds.any():
            raise ValueError("A value in x_new is below the interpolation "
                                "range.")
        if self.bounds_error and above_bounds.any():
            raise ValueError("A value in x_new is above the interpolation "
                                "range.")

        out_of_bounds = np.logical_or(below_bounds, above_bounds)
        return out_of_bounds
            
    #---------------------------------------------------------------------------
    def __call__(self, x_new):
        """
        Return the interpolated value
        """
        if self.extrap: 
            if np.isscalar(x_new):
                return self.linear_extrap(x_new)*1.
            else:
                return np.array([self.linear_extrap(x) for x in x_new])
        else:
            return self._evaluate_spline(x_new)*1.
       
    #--------------------------------------------------------------------------- 
    def _evaluate_spline(self, x_new):
        """
        Evaluate the spline
        """
        out_of_bounds = self._check_bounds(x_new)
        y_new = InterpolatedUnivariateSpline.__call__(self, x_new)
        if np.isscalar(y_new) or y_new.ndim == 0:
            return self.fill_value if out_of_bounds else y_new
        else:
            y_new[out_of_bounds] = self.fill_value
            return y_new
    
    #---------------------------------------------------------------------------
    def linear_extrap(self, x):
        """
        Do a linear extrapolation
        """
        if x < self.x[0]:
            return self.y[0] + (x-self.x[0])*(self.y[1]-self.y[0])/(self.x[1]-self.x[0])
        elif x > self.x[-1]:
            return self.y[-1] + (x-self.x[-1])*(self.y[-1]-self.y[-2])/(self.x[-1]-self.x[-2])
        else:
            return self._evaluate_spline(x)
    
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
# BIAS TO MASS RELATION
#-------------------------------------------------------------------------------
class BiasToMassRelation(Cache):
    """
    Class to handle conversions between mass (in M_sun/h) and bias quickly 
    using an interpolation table
    """
    # define the interpolation grid
    interpolation_grid = {}
    interpolation_grid['sigma8_z'] = np.linspace(0.3, 1.0, 100)
    interpolation_grid['b1'] = np.linspace(0.9, 7., 50)
    
    #---------------------------------------------------------------------------
    def __init__(self, z, cosmo, interpolated=False):
        """
        Parameters
        ----------
        z : float
            The redshift to compute the relation at
        cosmo : pygcl.Cosmology
            The cosmology object
        interpolated : bool, optional
            Whether to return results from an interpolation table
        """
        # initialize the Cache base class
        super(BiasToMassRelation, self).__init__()
        
        # save the parameters
        self.z = z
        self.cosmo = cosmo
        self.interpolated = interpolated
        self.delta_halo = 200
        
    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
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
        return val
        
    @parameter
    def cosmo(self, val):
        """
        The cosmology of the input linear power spectrum
        """
        return val
        
    @parameter
    def delta_halo(self, val):
        return val
            
            
    @cached_property("cosmo")
    def power_lin(self):
        """
        The `pygcl.LinearPS` object defining the linear power spectrum at `z=0`
        """
        return pygcl.LinearPS(self.cosmo, 0.)
        
    #---------------------------------------------------------------------------
    @cached_property()
    def interpolation_table(self):
        """
        Evaluate the bias to mass relation at the interpolation grid points
        """
        # setup a few functions we need
        mean_dens = self.cosmo.rho_bar_z(0.)
        mass_norm = 1e13
        mass_to_radius = lambda M: (3.*M*mass_norm/(4.*np.pi*mean_dens))**(1./3.)
        s8_0 = self.cosmo.sigma8()
        
        # setup the sigma spline
        R_interp = np.logspace(-3, 4, 1000)
        sigmas = self.power_lin.Sigma(R_interp)
        sigma_spline = RSDSpline(R_interp, sigmas, bounds_error=True)
        
        # the objective function to minimize
        def objective(mass, bias, rescaling):
            sigma = rescaling*sigma_spline(mass_to_radius(mass))
            return bias_Tinker(sigma) - bias
        
        # setup the grid points
        sigma8s = self.interpolation_grid['sigma8_z']
        b1s = self.interpolation_grid['b1']
        pts = np.asarray(list(itertools.product(sigma8s, b1s)))

        grid_vals = []
        for (s8_z, b1) in pts:
            
            # get appropriate rescalings
            rescaling = (s8_z / s8_0)
            
            # find the zero
            try:
                M = brentq(objective, 1e-10, 1e4, args=(b1, rescaling))*mass_norm
            except Exception as e:
                print s8, b1
                print objective(1e-10, b1, rescaling), objective(1e4, b1, rescaling)
                raise Exception(e)
                
            grid_vals.append(M)
        grid_vals = np.array(grid_vals).reshape((len(sigma8s), len(b1s)))
        
        # return the interpolator
        return RegularGridInterpolator((sigma8s, b1s), grid_vals)
        
    #---------------------------------------------------------------------------
    @unpacked
    def __call__(self, sigma8_z, b1):
        """
        Return the mass [units: `M_sun/h`] associated with the desired 
        `b1` and `sigma8` values
        
        Parameters
        ----------
        sigma8_z : float
            The sigma8 value
        b1 : float
            The linear bias
        """        
        if self.interpolated:
            return self.interpolation_table([sigma8_z, b1])
        else:
            
            # setup a few functions we need
            mean_dens = self.cosmo.rho_bar_z(0.)
            mass_norm = 1e13
            mass_to_radius = lambda M: (3.*M*mass_norm/(4.*np.pi*mean_dens))**(1./3.)
            rescaling = sigma8_z / self.cosmo.sigma8()
            
            # the objective function to minimize
            def objective(mass):
                sigma = rescaling*self.power_lin.Sigma(mass_to_radius(mass))
                return bias_Tinker(sigma, delta_halo=self.delta_halo) - b1

            return brentq(objective, 1e-8, 1e3)*mass_norm
    #---------------------------------------------------------------------------
            
#-------------------------------------------------------------------------------
# Sigma-Bias relation
#-------------------------------------------------------------------------------
class BiasToSigmaRelation(BiasToMassRelation):
    """
    Class to represent the relation between velocity dispersion and halo bias
    """
    def __init__(self, z, cosmo, interpolated=False, sigmav_0=None, M_0=None):
        """
        Parameters
        ----------
        z : float
            The redshift to compute the relation at
        cosmo : pygcl.Cosmology
            The cosmology object
        interpolated : bool, optional
            Whether to return results from an interpolation table
        """
        # initialize the base class
        super(BiasToSigmaRelation, self).__init__(z, cosmo, interpolated)
        
        # store the normalizations
        self.sigmav_0 = sigmav_0
        self.M_0 = M_0
        
        
    #---------------------------------------------------------------------------
    @parameter
    def sigmav_0(self, val):
        """
        The velocity normalization of the sigma to mass relation
        """
        return val
        
    @parameter
    def M_0(self, val):
        """
        The mass normalization of the sigma to mass relation
        """
        return val
    
    #---------------------------------------------------------------------------
    # Cached properties
    #---------------------------------------------------------------------------
    @cached_property("z", "cosmo")
    def Hz(self):
        """
        The Hubble parameter at `z`
        """
        return self.cosmo.H_z(self.z)
        
    @cached_property("z", "cosmo")
    def Ez(self):
        """
        The normalized Hubble parameter at `z`
        """
        return self.cosmo.H_z(self.z) / self.cosmo.H0()
        
    #---------------------------------------------------------------------------
    def evrard_sigmav(self, mass):
        """
        Return the line-of-sight velocity dispersion for dark matter using
        the relationship found between velocity dispersion and halo mass found
        by Evrard et al. 2008:
        
        \sigma_DM(M, z) = (1082.9 km/s) * (E(z) * M / 1e15 M_sun/h)**0.33
        
        Parameters
        ----------
        mass : float
            The halo mass in units of `M_sun/h`
        """
        # model params
        sigmav_0 = 1082.9 # in km/s
        M_0      = 1e15 # in M_sun/h
        alpha    = 0.336
        
        # the Evrard value
        sigmav = sigmav_0*(self.Ez * mass/M_0)**alpha
        
        # LOS is sigmav / sqrt(3) due to spherical symmetry
        sigmav /= 3**0.5
        
        # put it into units of Mpc/h
        return sigmav * (1 + self.z)/self.Hz
        
    #---------------------------------------------------------------------------
    @unpacked
    def mass(self, sigma8_z, b1):
        """
        Return the mass at this bias and sigma8 value
        """
        return BiasToMassRelation.__call__(self, sigma8_z, b1)
        
    @unpacked
    def __call__(self, sigma8_z, b1):
        """
        Return the velocity dispersion [units: `Mpc/h`] associated with the 
        desired `b1` and `sigma8` values
        
        Parameters
        ----------
        sigma8 : float
            The sigma8 value
        b1 : float
            The linear bias
        """
        # first get the halo mass
        M = self.mass(sigma8_z, b1)
    
        # evrard relation
        if self.sigmav_0 is None or self.M_0 is None:
            return self.evrard_sigmav(M)
        else:
            return self.sigmav_0 * (M / self.M_0)**(1./3.)
        
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------        
def mass_from_bias(bias, z, linearPS):
    """
    Given an input bias, return the corresponding mass, using Tinker et al.
    bias fits
    
    Notes
    -----
    This returns the halo mass in units of M_sun / h
    """
    # convenience normalization for finding the zero crossing
    mass_norm = 1e13
    
    # critical density in units of h^2 M_sun / Mpc^3
    kms_Mpc = pygcl.Constants.km/pygcl.Constants.second/pygcl.Constants.Mpc
    crit_dens = 3*(pygcl.Constants.H_0*kms_Mpc)**2 / (8*np.pi*pygcl.Constants.G) 
    
    unit_conversion = (pygcl.Constants.M_sun/pygcl.Constants.Mpc**3)
    crit_dens /= unit_conversion
    
    # mean density at z = 0
    cosmo = linearPS.GetCosmology()
    mean_dens = crit_dens * cosmo.Omega0_m()
    
    # convert mass to radius
    mass_to_radius = lambda M: (3.*M*mass_norm/(4.*np.pi*mean_dens))**(1./3.)
    
    # growth factor at this z
    z_Plin = linearPS.GetRedshift()
    Dz = 1.
    if z_Plin != z: 
        Dz = (cosmo.D_z(z) / cosmo.D_z(z_Plin))

    def objective(mass):
        sigma = Dz*linearPS.Sigma(mass_to_radius(mass))
        return bias_Tinker(sigma) - bias
        
    return brentq(objective, 1e-5, 1e5)*mass_norm

#-------------------------------------------------------------------------------
def bias_Tinker(sigmas, delta_c=1.686, delta_halo=200):
    """
    Return the halo bias for the Tinker form.
    
    Tinker, J., et al., 2010. ApJ 724, 878-886.
    http://iopscience.iop.org/0004-637X/724/2/878
    """
    y = np.log10(delta_halo)
    
    # get the parameters as a function of halo overdensity
    A = 1. + 0.24*y*np.exp(-(4./y)**4)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4)
    c = 2.4
    
    nu = delta_c / sigmas
    return 1. - A * (nu**a)/(nu**a + delta_c**a) + B*nu**b + C*nu**c

#-------------------------------------------------------------------------------
def sigma_from_bias(bias, z, linearPS):
    """
    Return sigma from bias
    """
    # normalized Teppei's sims at z = 0.509 and sigma_sAsA = 3.6
    sigma0 = 3.6 # in Mpc/h
    M0 = 5.4903e13 # in M_sun / h
    return sigma0 * (mass_from_bias(bias, z, linearPS) / M0)**(1./3)
    
#-------------------------------------------------------------------------------
 

    
