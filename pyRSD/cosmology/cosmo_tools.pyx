#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
"""
 cosmo_tools.pyx
 cosmology: store the core cosmology tools here
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/23/2014
"""
import numpy as np
cimport numpy as np
from .cosmo import Cosmology

#-------------------------------------------------------------------------------
cpdef vectorize(object x):
    return np.array(x, copy=False, ndmin=1)
#end vectorize

#-------------------------------------------------------------------------------
cpdef H(object z, object params='Planck1_lens_WP_highL'):
    """
    The value of the Hubble constant at redshift z in km/s/Mpc
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    return params.H0*E(z, params=params)
#end H

#-------------------------------------------------------------------------------
cpdef E(object z, object params='Planck1_lens_WP_highL'):
    """
    The unitless Hubble expansion rate at redshift z, 
    modified to include non-constant w parameterized linearly 
    with z ( w = w0 + w1*z )

    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    z = vectorize(z)
    if not isinstance(params, Cosmology):
        params = Cosmology(params)

    return np.sqrt(params.omegam*(1.+z)**3 + params.omegar*(1.+z)**4 \
                    + params.omegak*(1.+z)**2 \
                    + params.omegal*(1.+ z)**(3.*(1+params.w)))
#end E

#-------------------------------------------------------------------------------
cpdef omega_m_z(object z, object params='Planck1_lens_WP_highL'):
    """
    The matter density omega_m as a function of redshift

    From Lahav et al. 1991 equations 11b-c. This is equivalent to 
    equation 10 of Eisenstein & Hu 1999.
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.   
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    return params.omegam*(1.+z)**3/E(z, params=params)**2
#end omega_m_z

#-------------------------------------------------------------------------------
cpdef omega_l_z(object z, object params='Planck1_lens_WP_highL'):
    """
    The dark energy density omega_l as a function of redshift
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)   
    return params.omegal/E(z, params=params)**2
#end omega_l_z

#-------------------------------------------------------------------------------
cpdef mass_to_radius(M, mean_dens):
    """
    Calculate radius of a region of space from its mass.
    
    Parameters
    ----------
    M : {float, np.ndarray}
        the masses
        
    mean_dens : float
        the mean density of the universe
        
    Returns
    ------
    R : {float, np.ndarray}
        The corresponding radii to M
    """
    return (3.*M/(4.*np.pi*mean_dens))**(1./3.)
#end mass_to_radius

#-------------------------------------------------------------------------------
cpdef radius_to_mass(R, mean_dens):
    """
    Calculates mass of a region of space from its radius

    Parameters
    ----------
    R : {float, np.ndarray}
        the radii

    mean_dens : float
        the mean density of the universe

    Returns
    ------
    M : {float, np.ndarray}
        The masses corresponding to the radii
    """
    return 4*np.pi*R**3*mean_dens/3
#end radius_to_mass

#------------------------------------------------------------------------------- 
