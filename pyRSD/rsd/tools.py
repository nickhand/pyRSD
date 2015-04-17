from .. import pygcl, numpy as np

import bisect
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy.optimize import brentq
import warnings
import functools

import pandas as pd
from sklearn.gaussian_process import GaussianProcess

warnings.filterwarnings("ignore", category=DeprecationWarning,module="scipy")

#-------------------------------------------------------------------------------
class InterpolatedClass(object):
    """
    Class to handle interpolation lookup tables for a class
    """
    def __init__(self, index):
        
        self.index = index
        
    def make_table(self):
        
        self.table = np.empty(len(self.index), dtype=object)
        for i in self.index:
            self.update_index(i)
            self.zeldovich_power_table[sigma8] = spline(K_SPLINE, P00(K_SPLINE))
        

#-------------------------------------------------------------------------------
class RSDSpline(InterpolatedUnivariateSpline):
    """
    Class to implement a spline that remembers the x-domain
    """
    def __init__(self, *args, **kwargs):
        
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
            return self.linear_extrap(x_new)*1.
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
def monopole(f):
    """
    Decorator to compute the monopole from a `self.power` function
    """     
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, mus, **kwargs)
        return np.array([simps(Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k_obs))])
    return wrapper

#-------------------------------------------------------------------------------
def quadrupole(f):
    """
    Decorator to compute the quadrupole from a `self.power` function
    """     
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, mus, **kwargs)
        kern = 2.5*(3*mus**2 - 1.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k_obs))])
    return wrapper

#-------------------------------------------------------------------------------
def hexadecapole(f):
    """
    Decorator to compute the hexadecapole from a `self.power` function
    """  
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 1001)
        Pkmus = f(self, mus, **kwargs)
        kern = 9./8.*(35*mus**4 - 30.*mus**2 + 3.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k_obs))])
    return wrapper
    
#-------------------------------------------------------------------------------
# Sigma-Bias relation
#-------------------------------------------------------------------------------
class SigmaBiasRelation(object):
    """
    Class to represent the relation between velocity dispersion and halo bias
    """
    def __init__(self, z, linearPS):
        """
        Initialize and setup the splines
        """
        self.z = z
        self.power_lin = linearPS
        self._initialize_splines()
        
    #---------------------------------------------------------------------------
    def _initialize_splines(self):
        """
        Initialize the splines we need
        """
        biases = np.linspace(1.0, 7.0, 100)
        sigmas = np.array([sigma_from_bias(bias, self.z, self.power_lin) for bias in biases])
        
        self.sigma_to_bias_spline = RSDSpline(sigmas, biases, extrap=True)
        self.bias_to_sigma_spline = RSDSpline(biases, sigmas, extrap=True)
        
    #---------------------------------------------------------------------------
    def bias(self, sigma):
        """
        Return the linear bias for the input sigma in Mpc/h
        """
        return self.sigma_to_bias_spline(sigma)
    
    #-------------------------------------------------------------------------------
    def sigma(self, bias):
        """
        Return the sigma in Mpc/h for the input linear bias value
        """
        return self.bias_to_sigma_spline(bias)
    
    #-------------------------------------------------------------------------------
#endclass SigmaBiasRelation

#-------------------------------------------------------------------------------        
def mass_from_bias(bias, z, linearPS):
    """
    Given an input bias, return the corresponding mass, using Tinker et al.
    bias fits
    """
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
#end mass_from_bias

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
#end bias_Tinker

#-------------------------------------------------------------------------------
def sigma_from_bias(bias, z, linearPS):
    """
    Return sigma from bias
    """
    # normalized Teppei's sims at z = 0.509 and sigma_sAsA = 3.6
    sigma0 = 3.6 # in Mpc/h
    M0 = 5.4903e13 # in M_sun / h
    return sigma0 * (mass_from_bias(bias, z, linearPS) / M0)**(1./3)
 
#end sigma_from_bias

#-------------------------------------------------------------------------------  
def unpacked(method):
    @functools.wraps(method)
    def _decorator(*args):
        result = method(*args)
        return result if len(result) != 1 else result[0]
    return _decorator
    
    
