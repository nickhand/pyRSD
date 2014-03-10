#!python
#cython: boundscheck=False
#cython: wraparound=False
"""
 linear_growth.pyx
 pyPT: module to compute quantities related to the linear growth of perturbations
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/21/2014
"""
from .cosmo import Cosmology
from pyPT.cosmology cimport cosmo_tools

import numpy as np
cimport numpy as np


cpdef growth_function(object z, bint normed=True, object params="Planck1_lens_WP_highL"):
    """
    The linear growth function, defined as:

    .. math::  D(a) = 5/2 \Omega_{m, 0} H(a)/H0 \int_0^a 1/(a*H/H0)^3.

    If ``normed`` = ``True``, the returned value is normalized to unity at z = 0.

    Parameters
    ----------
    z : {float, np.ndarray}
        Redshift(s) to compute the growth function at.
    normed : bool, optional
        If ``True``, normalize to unity at z = 0. Default is ``True``.
    params : {str, dict, cosmo.Cosmology}
        The cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    z = cosmo_tools.vectorize(z)
    
    cdef np.ndarray zarr
    cdef int N = z.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    
    zarr = np.ascontiguousarray(z, dtype=np.double)
    output = np.ascontiguousarray(output, dtype=np.double)
    
    # set the cosmological parameters
    set_parameters(params.omegam, params.omegab, params.omegal, params.omegar, 
                    params.sigma_8, params.h, params.n, params.Tcmb, params.w, 0);
    
    # compute the growth function
    D_plus(<double *>zarr.data, N, int(normed), <double *>output.data)
    return output
#end growth_function

#-------------------------------------------------------------------------------
cpdef growth_rate(object z, object params='Planck1_lens_WP_highL'):
    """
    The growth rate, which is the logarithmic derivative of the growth 
    function with respect to scale factor, denoted by f usually. Fitting formula 
    from eq 5 of Hamilton 2001; originally from Lahav et al. 1991.
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    Ez = cosmo_tools.E(z, params=params)
    Dplus = growth_function(z, normed=False, params=params)

    return 0.5*((1+z)/Ez)**2*(5*params.omegam/Dplus - 3*params.omegam*(1+z) - 2*params.omegak)
#end growth_rate

#-------------------------------------------------------------------------------
cpdef Pk_lin(k_hMpc, z, tf="EH", params='Planck1_lens_WP_highL'):
    """
    Compute the linear matter power spectrum using the specified transfer
    function, appropriately normalized via sigma_8, at redshift z. 
    The primordial spectrum is assumed to be proportional to k^n.

    Parameters
    ----------
    k_hMpc : {float, np.ndarray}
        the wavenumber in units of h / Mpc
    z : float
        the redshift to compute the spectrum at
    tf : {``EH``, ``EH_no_wiggles``, ``EH_no_baryons``, ``BBKS``, ``Bond_Efs``}, optional
        The transfer to use. Default is the full CDM+baryon Eisenstein + Hu function.
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.     
    
    Returns
    -------
    P_k : numpy.ndarray
        linear power spectrum in units of Mpc^3/h^3
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    k = cosmo_tools.vectorize(k_hMpc)
    
    cdef int transfer = 0
    cdef np.ndarray karr
    cdef int N = k.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    
    # determine the transfer function number
    if tf == 'EH': 
        transfer = 0
    elif tf == 'EH_no_wiggles': 
        transfer = 1
    elif tf == 'EH_no_baryons':
        transfer = 2
    elif tf == 'BBKS': 
        transfer = 3
    elif tf == 'Bond_Efs': 
        transfer = 4
            
    # set the cosmological parameters
    set_parameters(params.omegam, params.omegab, params.omegal, params.omegar, 
                    params.sigma_8, params.h, params.n, params.Tcmb, params.w, transfer)
    
    # set up the arrays to pass to the C code
    karr = np.ascontiguousarray(k, dtype=np.double)
    output = np.ascontiguousarray(output, dtype=np.double)
    
    # compute the power spectrum at z
    linear_power(<double *>karr.data, z, N, <double *>output.data)
    
    return output
#end Pk_lin

#-------------------------------------------------------------------------------
cpdef Pk_nonlin(k_hMpc, z, tf='EH', params='Planck1_lens_WP_highL'):
    """
    Compute the nonlinear matter power spectrum using Halofit prescription
    from Smith et al. 2003, with updated fitting formulas from 
    Takahashi et al. 2012. It is appropriately normalized via sigma_8, 
    at redshift z. 

    Parameters
    ----------
    k_hMpc : {float, np.ndarray}
        the wavenumber in units of h / Mpc
    z : float
        the redshift to compute the spectrum at
    tf : {``EH``, ``EH_no_wiggles``, ``EH_no_baryons``, ``BBKS``, ``Bond_Efs``}, optional
        The linear transfer to use. Default is the full 
        CDM+baryon Eisenstein + Hu function.
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.     
    
    Returns
    -------
    P_k : numpy.ndarray
        linear power spectrum in units of Mpc^3/h^3
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    k = cosmo_tools.vectorize(k_hMpc)
    
    cdef int transfer = 0
    cdef np.ndarray karr
    cdef int N = k.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    
    # determine the transfer function number
    if tf == 'EH': 
        transfer = 0
    elif tf == 'EH_no_wiggles': 
        transfer = 1
    elif tf == 'EH_no_baryons':
        transfer = 2
    elif tf == 'BBKS': 
        transfer = 3
    elif tf == 'Bond_Efs': 
        transfer = 4
            
    # set the cosmological parameters
    set_parameters(params.omegam, params.omegab, params.omegal, params.omegar, 
                    params.sigma_8, params.h, params.n, params.Tcmb, params.w, transfer)
    
    # set up the arrays to pass to the C code
    karr = np.ascontiguousarray(k, dtype=np.double)
    output = np.ascontiguousarray(output, dtype=np.double)
    
    # compute the power spectrum at z
    nonlinear_power(<double *>karr.data, z, N, <double *>output.data)
    
    return output
#end Pk_nonlin


#-------------------------------------------------------------------------------
cpdef mass_variance(object r_Mpch, object z, bint normed=True, 
            object tf="EH", object params="Planck1_lens_WP_highL"):
    """
    The average mass fluctuation within a sphere of radius r, using the specified
    transfer function for the linear power spectrum.
    
    Parameters
    ----------
    r_Mpch : {float, np.ndarray}
        the radius to compute the variance within in units of Mpc/h
    z : {float, np.ndarray}
        the redshift to compute the variance at
    normed : bool, optional
        whether to normalize to the present value of sigma_8, the mass fluctuation
        within a sphere of radius 8 Mpc/h. Default is True
    tf : {``EH``, ``EH_no_wiggles``, ``EH_no_baryons``, ``BBKS``, ``Bond_Efst``}, optional
        The transfer to use. Default is the full CDM+baryon Eisenstein + Hu function.
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    # check that both r_Mpc or z are not arrays
    if not np.isscalar(r_Mpch) and not np.isscalar(z):
        raise ValueError("Radius and redshift inputs cannot both be arrays")
    
    # make cosmo parameter dictionary
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    # vectorize the radius input
    r = cosmo_tools.vectorize(r_Mpch)
    
    cdef int transfer = 0
    cdef np.ndarray rarr
    cdef int N = r.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    
    # determine the transfer function number
    if tf == 'EH': 
        transfer = 0
    elif tf == 'EH_no_wiggles': 
        transfer = 1
    elif tf == 'EH_no_baryons':
        transfer = 2
    elif tf == 'BBKS': 
        transfer = 3
    elif tf == 'Bond_Efs': 
        transfer = 4
    
    # set the cosmological parameters
    set_parameters(params.omegam, params.omegab, params.omegal, params.omegar, 
                    params.sigma_8, params.h, params.n, params.Tcmb, params.w, transfer)
    
    
    # set up the arrays to pass to the C code
    rarr = np.ascontiguousarray(r, dtype=np.double)
    output = np.ascontiguousarray(output, dtype=np.double)
    
    # compute the growth function
    Dz = growth_function(z, normed=True, params=params)
    
    # compute sigma at Z
    sigma_r(<double *>rarr.data, 0., N, 1, <double *>output.data)
    
    return Dz*output
#end mass_variance

#-------------------------------------------------------------------------------
cpdef dlnsdlnm(object r_Mpch, object sigma0=None, 
                object tf="EH", object params="Planck1_lens_WP_highL"):
    """
    The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln M}\right|`, ``len=len(r_Mpch)``
    For use in computing halo mass functions.

    Notes
    -----

    .. math:: frac{d\ln\sigma}{d\ln M} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk
    """
    
    # make cosmo parameter dictionary
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    # vectorize the radius input
    r = cosmo_tools.vectorize(r_Mpch)
    
    cdef int transfer = 0
    cdef np.ndarray rarr
    cdef int N = r.shape[0]
    cdef np.ndarray[double, ndim=1] integral = np.empty(N)
    
    # determine the transfer function number
    if tf == 'EH': 
        transfer = 0
    elif tf == 'EH_no_wiggles': 
        transfer = 1
    elif tf == 'EH_no_baryons':
        transfer = 2
    elif tf == 'BBKS': 
        transfer = 3
    elif tf == 'Bond_Efs': 
        transfer = 4
    
    # set the cosmological parameters
    set_parameters(params.omegam, params.omegab, params.omegal, params.omegar, 
                    params.sigma_8, params.h, params.n, params.Tcmb, params.w, transfer)
    
    
    # set up the arrays to pass to the C code
    rarr = np.ascontiguousarray(r, dtype=np.double)
    integral = np.ascontiguousarray(integral, dtype=np.double)

    # compute the mass variance if it is not provided
    if sigma0 is None:
        sigma0 = mass_variance(r_Mpch, 0., normed=True, tf=tf, params=params)

    # compute the integral, (unnormalized)
    dlnsdlnm_integral(<double *>rarr.data, N, <double *>integral.data)
    
    # now compute the full derivative
    dlnsdlnm = 3./(2*sigma0**2*np.pi**2*r**4)*integral
    
    return dlnsdlnm
#end dlnsdlnm

#-------------------------------------------------------------------------------