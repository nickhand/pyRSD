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
from cosmo import Cosmology
cimport cosmo_tools
cimport gsl_tools

import numpy as np
cimport numpy as np

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

#---------------------------------------------------------------------------
cpdef growth_function(object z, bint normed=True, object params="Planck1_lens_WP_highL"):
    r"""
    The linear growth function, defined as
    
    D(a) = 5/2 \Omega_{m, 0} H(a)/H0 \int_0^a 1/(a*H/H0)^3.
    
    If normed = True, the returned value is normalized to unity at z = 0.
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    normed : bool, default = True
        If True, normalize to unity at z = 0
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    z = cosmo_tools.vectorize(z)
    
    D = gsl_tools.compute_growth_integral(z, params)
    Ez = cosmo_tools.E(z, params=params) 
    D = 5./2*params.omegam*Ez*D
    
    norm = 1.
    if normed:
        norm = growth_function(0., normed=False, params=params)
    return D/norm
#end growth_function
 
#-------------------------------------------------------------------------------
cpdef Pk_full(object k_hMpc, object z, object params='Planck1_lens_WP_highL'):
    """
    Compute the CDM + baryon linear power spectrum using the full 
    Eisenstein and Hu transfer function fit, appropriately normalized 
    via sigma_8, at redshift z. The primordial spectrum is assumed 
    to be proportional to k^n.
    
    Calls the function TFfit_onek() from the tf_fit.c code.

    Parameters
    ----------
    k_hMpc : {float, np.ndarray}
        the wavenumber in units of h / Mpc
    z : float
        the redshift to compute the spectrum at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.     
    
    Returns
    -------
    P_k : numpy.ndarray
        full, linear power spectrum in units of Mpc^3/h^3
    """
    cdef float baryon_piece, cdm_piece
    
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    k = cosmo_tools.vectorize(k_hMpc)
    
    # initialize the transfer function parameters
    TFset_parameters(params.omegam*params.h*params.h, params.omegab/params.omegam, params.Tcmb)
    
    # compute the power spectrum normalization
    norm = (mass_variance(8., 0., normed=False, tf="full", params=params)/params.sigma_8)**2
    
    Tk = Tk_full(k, params.h)
    Dz = growth_function(z, params=params)
    return k**params.n * (Tk*Dz)**2 / norm
#end Pk_full

#---------------------------------------------------------------------------
cpdef Pk_nowiggles(object k_hMpc, object z, object params='Planck1_lens_WP_highL'):
    """
    Compute the CDM + baryon linear power spectrum with no oscillatory 
    features using the "no-wiggles" Eisenstein and Hu transfer function fit, 
    appropriately normalized via sigma_8, at redshift z. The primordial 
    spectrum is assumed to be proportional to k^n.
    
    Calls the function TFnowiggles() from the tf_fit.c code.

    Parameters
    ----------
    k_hMpc : {float, np.ndarray}
        the wavenumber in units of h / Mpc
    z : float
        the redshift to compute the spectrum at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    
    Returns
    -------
    P_k : numpy.ndarray
        no-wiggle, linear power spectrum in units of Mpc^3/h^3
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)      
    # if not all(k in params.keys() for k in ['omegam', 'omegab', 'Tcmb', 'h']):
    #     raise ValueError("Cannot compute no-wiggle power spectrum with given cosmology")
        
    k = cosmo_tools.vectorize(k_hMpc)

    # initialize the transfer function parameters
    TFset_parameters(params.omegam*params.h*params.h, params.omegab/params.omegam, params.Tcmb)

    # compute the power spectrum normalization
    norm  = (mass_variance(8., 0., normed=False, tf="no-wiggle", params=params)/params.sigma_8)**2
    
    fb = params.omegab/params.omegam
    Tk = Tk_nowiggles(k, params.omegam, fb, params.h, params.Tcmb) 
    Dz = growth_function(z, params=params)
    return k**params.n * (Tk*Dz)**2 / norm
#end Pk_nowiggles

#---------------------------------------------------------------------------
cpdef np.ndarray Tk_full(np.ndarray k, double h):
    """
    Wrapper function to compute the full Eisenstein + Hu tranfer function. Calls
    TFfit_onek() from tf_fit.c.
    
    Parameters
    ----------
    k : np.ndarray
        the wavenumbers in units of h/Mpc
    h : float
        the dimensionless Hubble parameter
    """
    cdef float baryon_piece, cdm_piece, this_Tk
    cdef int N = k.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    cdef int i
    
    for i in xrange(N):
        this_Tk = TFfit_onek(k[i]*h, &baryon_piece, &cdm_piece)
        output[i] = <double>this_Tk   
    return output
#end _Tk_full

#---------------------------------------------------------------------------
cpdef np.ndarray Tk_nowiggles(np.ndarray k, double omegam, double fb, double h, double Tcmb):
    """
    Wrapper function to compute the no-wiggle Eisenstein + Hu tranfer function. 
    Calls TFnowiggles() from tf_fit.c.
    
    Parameters
    ----------
    k : np.ndarray
        the wavenumbers in units of h/Mpc
    h : float
        the dimensionless Hubble parameter 
    """
    cdef float this_Tk
    cdef int N = k.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    cdef int i

    for i in xrange(N):
        this_Tk = TFnowiggles(omegam, fb, h, Tcmb, k[i])
        output[i] = <double>this_Tk
    return output
#end _Tk_nowiggles
#-------------------------------------------------------------------------------

cpdef mass_variance(object r_Mpch, object z, bint normed=True, 
            object tf="full", object params="Planck1_lens_WP_highL"):
    """
    The average mass fluctuation within a sphere of radius r, using the 
    CDM + baryon linear power spectrum and the Eisenstein + Hu transfer function,
    which is smoothed by a top hat function of the input radis
    
    Parameters
    ----------
    r_Mpch : {float, np.ndarray}
        the radius to compute the variance within in units of Mpc/h
    z : {float, np.ndarray}
        the redshift to compute the variance at
    normed : bool, optional
        whether to normalize to the present value of sigma_8, the mass fluctuation
        within a sphere of radius 8 Mpc/h. Default is True
    tf : str, optional
        the transfer to use, either "full" or "no-wiggle". Default is "full"
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
    
    # compute the variance at z=0
    sigma0 = gsl_tools.compute_sigma_integral(r, params, tf)
    
    # return sigma with the proper normalization
    Dz = growth_function(z, params=params)
    norm = 1.
    if normed: 
        norm = mass_variance(8., 0., normed=False, params=params)/params.sigma_8
    return np.sqrt(sigma0*Dz*Dz)/norm
#end mass_variance
#-------------------------------------------------------------------------------
    
