#!python
#cython: cdivision=True
#cython: boundscheck=False
#cython: wraparound=False
"""
 power_1loop.pyx
 pyPT: 1 loop density/velocity spectra are computed here
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/04/2014
"""
from pyPT.power cimport integralsIJ
from pyPT.cosmology cimport cosmo_tools, growth 
from ..cosmology import cosmo

import numpy as np
cimport numpy as np

KMAX = 1e3
KMIN = 1e-7
CONVERGENCE_FACTOR = 100.

#-------------------------------------------------------------------------------   
cpdef Pdd_1loop(k_hMpc, z, kmin=KMIN, kmax=KMAX, num_threads=1, 
                cosmo='Planck1_lens_WP_highL', return_components=False):
    """
    The 1-loop auto-correlation of dark matter density.
    
    Parameters
    ----------
    k_hMpc : {float, array_like}
        The wavenumbers to compute the power at in units of ``h/Mpc``.
    z : float
        The redshift to compute the power spectrum at.
    kmin : float, optional
        The minimum wavenumber to integrate over. Default is ```1e-7 h/Mpc``.
    kmax : float, optional
        The maximum wavenumber to integrate over. Default is ```1000 h/Mpc``.
    num_threads : int, optional
        The number of threads to use, default is ``1``.
    cosmo : {str, dict, cosmo.Cosmology}
        The cosmological parameters to use. Default is Planck DR1 + lensing
        + WP + high L 2013 parameters.
    return_components : bool, optional
        Whether to return the redshift-independent components. Default is 
        ``False``.
    """
    # initialize the wavenumber array
    k = np.array(k_hMpc, copy=False, ndmin=1)
    
    # make sure the wavenumbers are within the right range, given kmin/kmax
    if np.amin(k) < CONVERGENCE_FACTOR*kmin:
        raise ValueError("With kmin = %e h/Mpc, power spectrum cannot be computed " %kmin + \
                        "for wavenumbers less than %.2e h/Mpc" %(kmin*CONVERGENCE_FACTOR))
    if np.amax(k) > kmax/CONVERGENCE_FACTOR:
        raise ValueError("With kmax = %e h/Mpc, power spectrum cannot be computed " %kmax + \
                        "for wavenumbers greater than %.2e h/Mpc" %(kmax/CONVERGENCE_FACTOR))
    
    # compute the linear power spectrum over wide range of k for interpolation
    klin = np.logspace(np.log(1e-8), np.log(1e5), 10000, base=np.e)
    Plin = growth.Pk_lin(klin, 0., tf="EH", params=cosmo)
    
    # compute the Inm integrals in parallel
    I = integralsIJ.I_nm(0, 0, klin, Plin, k2=None, P2=None)
    I00 = I.evaluate(k, kmin, kmax, num_threads)

    # compute the Jnm integrals too
    J = integralsIJ.J_nm(0, 0, klin, Plin)
    J00 = J.evaluate(k, kmin, kmax)
    
    # the velocity and growth factors
    D = growth.growth_function(z, normed=True, params=cosmo)
     
    Plin = growth.Pk_lin(k, 0., tf='EH', params=cosmo)
    P11 = Plin
    P22 = 2.*I00
    P13 = 6.*k*k*J00*Plin
    
    if return_components:
        return P22, P13
    else:
        return D**2*P11 + D**4*(P22 + P13)
#end Pdd_1loop

#-------------------------------------------------------------------------------   
cpdef Pdv_1loop(k_hMpc, z, kmin=KMIN, kmax=KMAX, num_threads=1, 
                cosmo='Planck1_lens_WP_highL', return_components=False):
    """
    The 1-loop cross-correlation between dark matter density and velocity 
    divergence.
    
    Parameters
    ----------
    k_hMpc : {float, array_like}
        The wavenumbers to compute the power at in units of ``h/Mpc``.
    z : float
        The redshift to compute the power spectrum at.
    kmin : float, optional
        The minimum wavenumber to integrate over. Default is ```1e-7 h/Mpc``.
    kmax : float, optional
        The maximum wavenumber to integrate over. Default is ```1000 h/Mpc``.
    num_threads : int, optional
        The number of threads to use, default is ``1``.
    cosmo : {str, dict, cosmo.Cosmology}
        The cosmological parameters to use. Default is Planck DR1 + lensing
        + WP + high L 2013 parameters.
    return_components : bool, optional
        Whether to return the redshift-independent components. Default is 
        ``False``.
    """
    # initialize the wavenumber array
    k = np.array(k_hMpc, copy=False, ndmin=1)
    
    # make sure the wavenumbers are within the right range, given kmin/kmax
    if np.amin(k) < CONVERGENCE_FACTOR*kmin:
        raise ValueError("With kmin = %e h/Mpc, power spectrum cannot be computed " %kmin + \
                        "for wavenumbers less than %.2e h/Mpc" %(kmin*CONVERGENCE_FACTOR))
    if np.amax(k) > kmax/CONVERGENCE_FACTOR:
        raise ValueError("With kmax = %e h/Mpc, power spectrum cannot be computed " %kmax + \
                        "for wavenumbers greater than %.2e h/Mpc" %(kmax/CONVERGENCE_FACTOR))
    
    # compute the linear power spectrum over wide range of k for interpolation
    klin = np.logspace(np.log(1e-8), np.log(1e5), 10000, base=np.e)
    Plin = growth.Pk_lin(klin, 0., tf='EH', params=cosmo)
    
    # compute the Inm integrals in parallel
    I = integralsIJ.I_nm(0, 1, klin, Plin, k2=None, P2=None)
    I01 = I.evaluate(k, kmin, kmax, num_threads)

    # compute the Jnm integrals too
    J = integralsIJ.J_nm(0, 1, klin, Plin)
    J01 = J.evaluate(k, kmin, kmax)
    
    # the velocity and growth factors
    D          = growth.growth_function(z, normed=True, params=cosmo)
    f          = growth.growth_rate(z, params=cosmo)
    conformalH = cosmo_tools.H(z,  params=cosmo)/(1.+z)
    A = -f*conformalH
    
    Plin = growth.Pk_lin(k, 0., tf='EH', params=cosmo)
    P11  = Plin
    P22  = 2.*I01
    P13  = 6.*k*k*Plin*J01

    if return_components:
        return P22, P13
    else:
        return A*(D**2*P11 + D**4*(P22 + P13))
#end Pdv_1loop

#-------------------------------------------------------------------------------
cpdef Pvv_1loop(k_hMpc, z, kmin=KMIN, kmax=KMAX, num_threads=1, 
                cosmo='Planck1_lens_WP_highL', return_components=False):
    """
    The 1-loop auto-correlation of dark matter velocity divergence.
    
    Parameters
    ----------
    k_hMpc : {float, array_like}
        The wavenumbers to compute the power at in units of ``h/Mpc``.
    z : float
        The redshift to compute the power spectrum at.
    kmin : float, optional
        The minimum wavenumber to integrate over. Default is ```1e-7 h/Mpc``.
    kmax : float, optional
        The maximum wavenumber to integrate over. Default is ```1000 h/Mpc``.
    num_threads : int, optional
        The number of threads to use, default is ``1``.
    cosmo : {str, dict, cosmo.Cosmology}
        The cosmological parameters to use. Default is Planck DR1 + lensing
        + WP + high L 2013 parameters.
    return_components : bool, optional
        Whether to return the redshift-independent components. Default is 
        ``False``.
    """   
    # initialize the wavenumber array
    k = np.array(k_hMpc, copy=False, ndmin=1)
    
    # make sure the wavenumbers are within the right range, given kmin/kmax
    if np.amin(k) < CONVERGENCE_FACTOR*kmin:
        raise ValueError("With kmin = %e h/Mpc, power spectrum cannot be computed " %kmin + \
                        "for wavenumbers less than %.2e h/Mpc" %(kmin*CONVERGENCE_FACTOR))
    if np.amax(k) > kmax/CONVERGENCE_FACTOR:
        raise ValueError("With kmax = %e h/Mpc, power spectrum cannot be computed " %kmax + \
                        "for wavenumbers greater than %.2e h/Mpc" %(kmax/CONVERGENCE_FACTOR))
    
    # compute the linear power spectrum over wide range of k for interpolation
    klin = np.logspace(np.log(1e-8), np.log(1e5), 10000, base=np.e)
    Plin = growth.Pk_lin(klin, 0., tf='EH', params=cosmo)

    # compute the Inm integrals in parallel
    I = integralsIJ.I_nm(1, 1, klin, Plin, k2=None, P2=None)
    I11 = I.evaluate(k, kmin, kmax, num_threads)

    # compute the Jnm integrals too
    J = integralsIJ.J_nm(1, 1, klin, Plin)
    J11 = J.evaluate(k, kmin, kmax)

    # the velocity and growth factors
    D          = growth.growth_function(z, normed=True, params=cosmo)
    f          = growth.growth_rate(z, params=cosmo)
    conformalH = cosmo_tools.H(z,  params=cosmo)/(1.+z)
    A = (f*conformalH)**2
    
    Plin = growth.Pk_lin(k, 0., tf='EH', params=cosmo) 
    P11  = Plin
    P22  = 2.*I11
    P13  = 6.*k*k*Plin*J11

    if return_components:
        return P22, P13
    else:
        return A*(D**2*P11 + D**4*(P22 + P13))
#end Pvv_1loop
#-------------------------------------------------------------------------------

