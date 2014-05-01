"""
 velocity.py
 pyRSD: functions for computing velocity dispersions in the halo model
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/27/2014
"""
from . import bias, hmf, cosmo_tools, cosmo
import scipy.integrate as intgr
import numpy as np

#-------------------------------------------------------------------------------
def sigma_evrard(M_Msunh, z, params):
    """
    Compute the small-scale dark matter halo velocity dispersion in km/s from 
    virial motions using the simulation results from Evrard et al. 2008. 
    
    Parameters
    ----------
    M_Msunh : {float, array_like}
        The halo mass in units of M_sun / h
    z : float
        The redshift to compute the velocity dispersion at. 
    params : {str, dict, cosmo.Cosmology}
        The cosmological parameters to use, specified by the name of a predefined
        cosmology, a parameter dictionary, or a Cosmology class.
    """
    sigma_DM = 1082.9 # normalization for M = 1e15 M_sun/h (in km/s)
    alpha    = 0.3361 # power law index
        
    if not isinstance(params, cosmo.Cosmology):
        params = cosmo.Cosmology(params)
    Ez = cosmo_tools.H(z, params=params)/params.H0
    return sigma_DM * (Ez*M_Msunh/1e15)**alpha
#end sigma_evrard

#-------------------------------------------------------------------------------
def sigma_v2(mf):
    """
    Compute the small-scale velocity dispersion, averaged over all halos, using
    the input hmf.HaloMassFunction object. 
    
    Parameters
    ----------
    mf : hmf.HaloMassFunction
        The object holding the relevant halo mass function and related quantities
        
    Notes
    -----
    .. math:: \sigma^2_{v^2} = \frac{1}{\bar{\rho}} \int dM M (dN/dM) v_{\parallel}^2  
    """
    def integrand(lnM):
        M = np.exp(lnM)
        return mf.dndlnm_spline(M)*M*sigma_evrard(M, mf.z, mf.cosmo)**2
        
    integral = intgr.quad(integrand, np.log(1e8), np.log(1e16), epsabs=0, epsrel=1e-4)[0]
    return np.sqrt(integral/mf.cosmo.mean_dens)
#end sigma_v2

#-------------------------------------------------------------------------------
def sigma_bv2(mf, bias_model):
    """
    Compute the small-scale velocity dispersion, weighted by halo bias and 
    averaged over all halos, using the input hmf.HaloMassFunction object.
    
    Parameters
    ----------
    mf : hmf.HaloMassFunction
        The object holding the relevant halo mass function and related quantities
    bias_model : {'Tinker', 'PS', 'SMT'}
        The name of the halo bias model to use, either Tinker, SMT or PS
        
    Notes
    -----
    .. math:: \sigma^2_{bv^2} = \frac{1}{\bar{\rho}} \int dM M (dN/dM) b(M) v_{\parallel}^2  
    """
    available_bias = ['Tinker', 'PS', 'SMT']
    if bias_model not in available_bias:
        raise ValueError("%s is an invalid bias model. Must be one %s" \
                            %(bias_model, available_bias))
    bias_args = (mf.delta_halo,) if bias_model == "Tinker" else ()
    bias_func = getattr(bias, 'bias_%s' %bias_model)
    
    def integrand(lnM):
        M = np.exp(lnM) # in units of M_sun/h
        R = cosmo_tools.mass_to_radius(M, mf.cosmo.mean_dens) # in Mpc/h
        sigma = mf._power.sigma_r(R, mf.z)
        b = bias_func(sigma, mf.delta_c, *bias_args)
        return mf.dndlnm_spline(M)*M*b*sigma_evrard(M, mf.z, mf.cosmo)**2
        
    integral = intgr.quad(integrand, np.log(1e8), np.log(1e16), epsabs=0, epsrel=1e-4)[0]
    return np.sqrt(integral/mf.cosmo.mean_dens)
#end sigma_bv2

#-------------------------------------------------------------------------------
def sigma_bv4(mf, bias_model):
    """
    Compute the small-scale velocity dispersion, weighted by halo bias and 
    averaged over all halos, using the input hmf.HaloMassFunction object.
    
    Parameters
    ----------
    mf : hmf.HaloMassFunction
        The object holding the relevant halo mass function and related quantities
    bias_model : {'Tinker', 'PS', 'SMT'}
        The name of the halo bias model to use, either Tinker, SMT or PS
        
    Notes
    -----
    .. math:: \sigma^2_{bv^2} = \frac{1}{\bar{\rho}} \int dM M (dN/dM) b(M) v_{\parallel}^4 
    """
    available_bias = ['Tinker', 'PS', 'SMT']
    if bias_model not in available_bias:
        raise ValueError("%s is an invalid bias model. Must be one %s" %(bias_model, available_bias))
    bias_args = (mf.delta_halo,) if bias_model == "Tinker" else ()
    bias_func = getattr(bias, 'bias_%s' %bias_model)
    
    def integrand(lnM):
        M = np.exp(lnM) # in units of M_sun/h
        R = cosmo_tools.mass_to_radius(M, mf.cosmo.mean_dens) # in Mpc/h
        sigma = mf._power.sigma_r(R, mf.z)
        b = bias_func(sigma, mf.delta_c, *bias_args)
        return mf.dndlnm_spline(M)*M*b*sigma_evrard(M, mf.z, mf.cosmo)**4
        
    integral = intgr.quad(integrand, np.log(1e8), np.log(1e16), epsabs=0, epsrel=1e-4)[0]
    return (integral/mf.cosmo.mean_dens)**(0.25)
#end sigma_bv4

#-------------------------------------------------------------------------------